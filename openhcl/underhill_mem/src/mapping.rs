// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// UNSAFETY: Implementing GuestMemoryAccess.
#![expect(unsafe_code)]

use crate::MemoryAcceptor;
use crate::MshvVtlWithPolicy;
use crate::RegistrationError;
use crate::registrar::MemoryRegistrar;
use guestmem::GuestMemoryAccess;
use guestmem::GuestMemoryBackingError;
use guestmem::PAGE_SIZE;
use hcl::ioctl::Mshv;
use hcl::ioctl::MshvVtlLow;
use hvdef::HvMapGpaFlags;
use inspect::Inspect;
use memory_range::MemoryRange;
use parking_lot::Mutex;
use sparse_mmap::SparseMapping;
use std::ptr::NonNull;
use std::sync::Arc;
use thiserror::Error;
use vm_topology::memory::MemoryLayout;

pub struct GuestPartitionMemoryView<'a> {
    memory_layout: &'a MemoryLayout,
    valid_memory: Arc<GuestValidMemory>,
}

impl<'a> GuestPartitionMemoryView<'a> {
    /// A bitmap is created to track the accessibility state of each page in the
    /// lower VTL memory. The bitmap is initialized to valid_bitmap_state.
    ///
    /// This is used to support tracking the shared/encrypted state of each
    /// page.
    pub fn new(
        memory_layout: &'a MemoryLayout,
        valid_bitmap_state: bool,
    ) -> Result<Self, MappingError> {
        let valid_memory =
            GuestValidMemory::new(memory_layout, valid_bitmap_state).map(Arc::new)?;
        Ok(Self {
            memory_layout,
            valid_memory,
        })
    }

    /// Returns the built partition-wide valid memory.
    pub fn partition_valid_memory(&self) -> Arc<GuestValidMemory> {
        self.valid_memory.clone()
    }

    /// Build a [`GuestMemoryMapping`], feeding in any related partition-wide
    /// state.
    fn build_guest_memory_mapping(
        &self,
        mshv_vtl_low: &MshvVtlLow,
        memory_mapping_builder: &mut GuestMemoryMappingBuilder,
    ) -> Result<GuestMemoryMapping, MappingError> {
        memory_mapping_builder
            .use_partition_valid_memory(Some(self.valid_memory.clone()))
            .build(mshv_vtl_low, self.memory_layout)
    }
}

/// Partition-wide (cross-vtl) tracking of valid memory that can be used in
/// individual GuestMemoryMappings.
#[derive(Debug)]
pub struct GuestValidMemory {
    valid_bitmap: GuestMemoryBitmap,
    valid_bitmap_lock: Mutex<()>,
}

impl GuestValidMemory {
    fn new(memory_layout: &MemoryLayout, valid_bitmap_state: bool) -> Result<Self, MappingError> {
        let valid_bitmap = {
            let mut bitmap = {
                // Calculate the total size of the address space by looking at the ending region.
                let last_entry = memory_layout
                    .ram()
                    .last()
                    .expect("memory map must have at least 1 entry");
                let address_space_size = last_entry.range.end();
                GuestMemoryBitmap::new(address_space_size as usize, GuestMemoryBitmapPageSize::PageSize4k)?
            };

            for entry in memory_layout.ram() {
                if entry.range.is_empty() {
                    continue;
                }

                bitmap.init(entry.range, valid_bitmap_state)?;
            }

            bitmap
        };

        Ok(GuestValidMemory {
            valid_bitmap,
            valid_bitmap_lock: Default::default(),
        })
    }

    /// Update the bitmap to reflect the validity of the given range.
    pub fn update_valid(&self, range: MemoryRange, state: bool) {
        let _lock = self.valid_bitmap_lock.lock();
        self.valid_bitmap.update(range, state);
    }

    /// Check if the given page is valid.
    pub(crate) fn check_valid(&self, gpn: u64) -> bool {
        self.valid_bitmap.page_state(gpn)
    }

    fn access_bitmap(&self) -> guestmem::BitmapInfo {
        let ptr = NonNull::new(self.valid_bitmap.as_ptr()).unwrap();
        guestmem::BitmapInfo {
            read_bitmap: ptr,
            write_bitmap: ptr,
            bit_offset: 0,
        }
    }
}

/// An implementation of a [`GuestMemoryAccess`] trait for Underhill VMs.
#[derive(Debug, Inspect)]
pub struct GuestMemoryMapping {
    #[inspect(skip)]
    mapping: SparseMapping,
    iova_offset: Option<u64>,
    #[inspect(with = "Option::is_some")]
    valid_memory: Option<Arc<GuestValidMemory>>,
    // TODO GUEST VSM: synchronize bitmap access
    #[inspect(with = "Option::is_some")]
    permission_bitmaps: Option<PermissionBitmaps>,
    #[inspect(with = "Option::is_some")]
    accepted_memory: Option<GuestAcceptedMemory>,
    registrar: Option<MemoryRegistrar<MshvVtlWithPolicy>>,
}

/// Bitmap implementation using sparse mapping that can be used to track page
/// states.
#[derive(Debug)]
struct PermissionBitmaps {
    permission_update_lock: Mutex<()>,
    read_bitmap: GuestMemoryBitmap,
    write_bitmap: GuestMemoryBitmap,
    kernel_execute_bitmap: GuestMemoryBitmap,
    user_execute_bitmap: GuestMemoryBitmap,
}

/// Bitmap to track acceptance status for lazy page acceptance.
#[derive(Debug)]
pub struct GuestAcceptedMemory {
    acceptance_lock: Mutex<()>,
    bitmap_lock: Mutex<()>,
    bitmap: GuestMemoryBitmap,
    acceptor: Arc<MemoryAcceptor>,
}

impl GuestAcceptedMemory {
    /// Check if the given page is accepted.
    pub fn check_acceptance(&self, gpn: u64) -> bool {
        self.bitmap.page_state(gpn)
    }

    /// Update the acceptance bitmap for the given range.
    pub fn update_acceptance(&self, range: MemoryRange, state: bool) {
        let _lock = self.bitmap_lock.lock();
        self.bitmap.update(range, state);
    }

    pub fn bitmap_page_size(&self) -> u64 {
        self.bitmap.page_size() as u64
    }    
}
#[derive(Error, Debug)]
pub enum VtlPermissionsError {
    #[error("no vtl 1 permissions enforcement, bitmap is not present")]
    NoPermissionsTracked,
}

#[derive(Debug)]
enum GuestMemoryBitmapPageSize {
    PageSize4k,
    PageSize2M,
}

impl GuestMemoryBitmapPageSize {
    #[inline(always)]
    /// Page size of each bit represented in the bitmap.
    fn bitmap_page_size(&self) -> usize {
        match self {
            GuestMemoryBitmapPageSize::PageSize4k => PAGE_SIZE as usize,
            GuestMemoryBitmapPageSize::PageSize2M => x86defs::X64_LARGE_PAGE_SIZE as usize,
        }
    }
}

#[derive(Debug)]
struct GuestMemoryBitmap {
    bitmap: SparseMapping,
    page_size: GuestMemoryBitmapPageSize,
}

impl GuestMemoryBitmap {
    fn new(
        address_space_size: usize,
        page_size: GuestMemoryBitmapPageSize
    ) -> Result<Self, MappingError> {
        let bitmap = 
            SparseMapping::new(
                (address_space_size / page_size.bitmap_page_size())
                .div_ceil(8)
            )
            .map_err(MappingError::BitmapReserve)?;
        bitmap
            .map_zero(0, bitmap.len())
            .map_err(MappingError::BitmapMap)?;
        Ok(Self { bitmap, page_size })
    }

    fn init(&mut self, range: MemoryRange, state: bool) -> Result<(), MappingError> {
        let page_size = self.page_size();
        if range.start() % page_size as u64 != 0 || range.end() % page_size as u64 != 0
        {
            return Err(MappingError::BadAlignment(range));
        }

        let bitmap_start = range.start() as usize / page_size / 8;
        let bitmap_end = (range.end() - 1) as usize / page_size / 8;
        let bitmap_page_start = bitmap_start / PAGE_SIZE;
        let bitmap_page_end = bitmap_end / PAGE_SIZE;
        let page_count = bitmap_page_end + 1 - bitmap_page_start;

        // TODO SNP: map some pre-reserved lower VTL memory into the
        // bitmap. Or just figure out how to hot add that memory to the
        // kernel. Or have the boot loader reserve it at boot time.
        self.bitmap
            .alloc(bitmap_page_start * PAGE_SIZE, page_count * PAGE_SIZE)
            .map_err(MappingError::BitmapAlloc)?;

        // Set the initial bitmap state.
        if state {
            self.update(range, true);
        }
        Ok(())
    }

    /// Panics if the range is outside of guest RAM.
    fn update(&self, range: MemoryRange, state: bool) {
        let page_size = self.page_size();
        assert!(
            range.start() % page_size as u64 == 0 && range.end() % page_size as u64 == 0,
            "range not aligned to bitmap page size",
        );
        let start_gpn = range.start() / page_size as u64;
        let end_gpn = range.end() / page_size as u64;
        if start_gpn >= end_gpn {
            return;
        }
        let update_bits = |byte_idx: u64, first_bit: u64, last_bit: u64| {
            let mut b = 0;
            self.bitmap
                .read_at(byte_idx as usize, std::slice::from_mut(&mut b))
                .unwrap();
            let mask = (((1u16 << (last_bit - first_bit + 1)) - 1) as u8) << first_bit;
            if state {
                b |= mask;
            } else {
                b &= !mask;
            }
            self.bitmap
                .write_at(byte_idx as usize, std::slice::from_ref(&b))
                .unwrap();
        };        

        let first_byte = start_gpn / 8;
        let last_byte = (end_gpn - 1) / 8;
        if first_byte == last_byte {
            update_bits(first_byte, start_gpn % 8, (end_gpn - 1) % 8);
            return;
        }

        // Handle updates to bitmap for pages not aligned to the bitmap 8-page boundary/e.
        if start_gpn % 8 != 0 {
             update_bits(first_byte, start_gpn % 8, 7);
        }
        if end_gpn % 8 != 0 {
            update_bits(last_byte, 0, (end_gpn - 1) % 8);
        }    

        // Handle updates to ranges aligned to the bitmap 8-page boundary.
        let aligned_start = if start_gpn % 8 == 0 { first_byte } else { first_byte + 1 };
        let aligned_end = if end_gpn % 8 == 0 { last_byte } else { last_byte.saturating_sub(1) };
        if aligned_end >= aligned_start {
            let fill_val = if state { 0xff } else { 0x00 };
            self.bitmap
                .fill_at(aligned_start as usize, fill_val, (aligned_end - aligned_start + 1) as usize)
                .unwrap();
        }
    }

    /// Read the bitmap for `gpn`.
    /// Panics if the range is outside of guest RAM.
    fn page_state(&self, gpn: u64) -> bool {
        let mut b = 0;
        self.bitmap
            .read_at(gpn as usize / 8, std::slice::from_mut(&mut b))
            .unwrap();
        b & (1 << (gpn % 8)) != 0
    }

    fn as_ptr(&self) -> *mut u8 {
        self.bitmap.as_ptr().cast()
    }

    pub fn page_size(&self) -> usize {
        self.page_size.bitmap_page_size()
    }
}

/// Error constructing a [`GuestMemoryMapping`].
#[derive(Debug, Error)]
pub enum MappingError {
    #[error("failed to allocate VA space for guest memory")]
    Reserve(#[source] std::io::Error),
    #[error("failed to map guest memory pages")]
    Map(#[source] std::io::Error),
    #[error("failed to allocate VA space for bitmap")]
    BitmapReserve(#[source] std::io::Error),
    #[error("failed to map zero pages for bitmap")]
    BitmapMap(#[source] std::io::Error),
    #[error("failed to allocate pages for bitmap")]
    BitmapAlloc(#[source] std::io::Error),
    #[error("memory map entry {0} has insufficient alignment to support a bitmap")]
    BadAlignment(MemoryRange),
    #[error("failed to open device")]
    OpenDevice(#[source] hcl::ioctl::Error),
}

/// A builder for [`GuestMemoryMapping`].
pub struct GuestMemoryMappingBuilder {
    physical_address_base: u64,
    valid_memory: Option<Arc<GuestValidMemory>>,
    permissions_bitmap_state: Option<bool>,
    acceptance_bitmap_state: Option<bool>,
    acceptor: Option<Arc<MemoryAcceptor>>,
    shared: bool,
    for_kernel_access: bool,
    dma_base_address: Option<u64>,
    ignore_registration_failure: bool,
}

impl GuestMemoryMappingBuilder {
    /// FUTURE: use bitmaps to track VTL permissions as well, to support guest
    /// VSM for hardware-isolated VMs.
    fn use_partition_valid_memory(
        &mut self,
        valid_memory: Option<Arc<GuestValidMemory>>,
    ) -> &mut Self {
        self.valid_memory = valid_memory;
        self
    }

    /// Set whether to allocate tracking bitmaps for memory access permissions,
    /// and specify the initial state of the bitmaps.
    ///
    /// This is used to support tracking the read/write/kernel execute/user
    /// execute permissions of each page.
    pub fn use_permissions_bitmaps(&mut self, initial_state: Option<bool>) -> &mut Self {
        self.permissions_bitmap_state = initial_state;
        self
    }

    /// Set whether to allocate tracking bitmaps for accept state for lazy acceptance of
    /// lower VTL pages and specify the initial state of the bitmaps.
    pub fn use_acceptance_bitmap(
        &mut self,
        acceptor: Option<Arc<MemoryAcceptor>>,
        initial_state: Option<bool>,
    ) -> &mut Self {
        assert!(
            initial_state.is_some() == acceptor.is_some(),
            "initial_state and acceptor must both be Some or both be None"
        );
        self.acceptance_bitmap_state = initial_state;
        self.acceptor = acceptor;
        self
    }

    /// Set whether this is a mapping to access shared memory.
    pub fn shared(&mut self, is_shared: bool) -> &mut Self {
        self.shared = is_shared;
        self
    }

    /// Set whether this mapping's memory can be locked to pass to the kernel.
    ///
    /// If so, then the memory will be registered with the kernel as part of
    /// `expose_va`, which is called when memory is locked.
    pub fn for_kernel_access(&mut self, for_kernel_access: bool) -> &mut Self {
        self.for_kernel_access = for_kernel_access;
        self
    }

    /// Sets the base address to use for DMAs to this memory.
    ///
    /// This may be `None` if DMA is not supported.
    ///
    /// The address to use depends on the backing technology. For SNP VMs, it
    /// should be either zero or the VTOM address, since shared memory is mapped
    /// twice. For TDX VMs, shared memory is only mapped once, but the IOMMU
    /// expects the SHARED bit to be set in DMA transactions, so it should be
    /// set here. And for non-isolated/software-isolated VMs, it should be zero
    /// or the VTL0 alias address, depending on which VTL this memory mapping is
    /// for.
    pub fn dma_base_address(&mut self, dma_base_address: Option<u64>) -> &mut Self {
        self.dma_base_address = dma_base_address;
        self
    }

    /// Ignore registration failures when registering memory with the kernel.
    ///
    /// This should be used when user mode is restarted for servicing but the
    /// kernel is not. Since this is not currently a production scenario, this
    /// is a simple way to avoid needing to track the state of the kernel
    /// registration across user-mode restarts.
    ///
    /// It is not a good idea to enable this otherwise, since the kernel very
    /// noisily complains if memory is registered twice, so we don't want that
    /// leaking into production scenarios.
    ///
    /// FUTURE: fix the kernel to silently succeed duplication registrations.
    pub fn ignore_registration_failure(&mut self, ignore: bool) -> &mut Self {
        self.ignore_registration_failure = ignore;
        self
    }

    /// Mapping should leverage the bitmap used to track the accessibility state
    /// of each page in the lower VTL memory.
    pub fn build_with_bitmap(
        &mut self,
        mshv_vtl_low: &MshvVtlLow,
        partition_builder: &GuestPartitionMemoryView<'_>,
    ) -> Result<GuestMemoryMapping, MappingError> {
        partition_builder.build_guest_memory_mapping(mshv_vtl_low, self)
    }

    pub fn build_without_bitmap(
        &self,
        mshv_vtl_low: &MshvVtlLow,
        memory_layout: &MemoryLayout,
    ) -> Result<GuestMemoryMapping, MappingError> {
        self.build(mshv_vtl_low, memory_layout)
    }

    /// Map the lower VTL address space.
    ///
    /// If `is_shared`, then map the kernel mapping as shared memory.
    ///
    /// Add in `file_starting_offset` to construct the page offset for each
    /// memory range. This can be the high bit to specify decrypted/shared
    /// memory, or it can be the VTL0 alias map start for non-isolated VMs.
    ///
    /// When handing out IOVAs for device DMA, add `iova_offset`. This can be
    /// VTOM for SNP-isolated VMs, or it can be the VTL0 alias map start for
    /// non-isolated VMs.
    fn build(
        &self,
        mshv_vtl_low: &MshvVtlLow,
        memory_layout: &MemoryLayout,
    ) -> Result<GuestMemoryMapping, MappingError> {
        // Calculate the file offset within the `mshv_vtl_low` file.
        let file_starting_offset = self.physical_address_base
            | if self.shared {
                MshvVtlLow::SHARED_MEMORY_FLAG
            } else {
                0
            };

        // Calculate the total size of the address space by looking at the ending region.
        let last_entry = memory_layout
            .ram()
            .last()
            .expect("memory map must have at least 1 entry");
        let address_space_size = last_entry.range.end();
        let mapping =
            SparseMapping::new(address_space_size as usize).map_err(MappingError::Reserve)?;

        tracing::trace!(?mapping, "map_lower_vtl_memory mapping");

        let mut permission_bitmaps = if self.permissions_bitmap_state.is_some() {
            Some(PermissionBitmaps {
                permission_update_lock: Default::default(),
                read_bitmap: GuestMemoryBitmap::new(address_space_size as usize, GuestMemoryBitmapPageSize::PageSize4k)?,
                write_bitmap: GuestMemoryBitmap::new(address_space_size as usize, GuestMemoryBitmapPageSize::PageSize4k)?,
                kernel_execute_bitmap: GuestMemoryBitmap::new(address_space_size as usize, GuestMemoryBitmapPageSize::PageSize4k)?,
                user_execute_bitmap: GuestMemoryBitmap::new(address_space_size as usize, GuestMemoryBitmapPageSize::PageSize4k)?,
            })
        } else {
            None
        };

        let mut accepted_memory = if self.acceptance_bitmap_state.is_some() {
            Some(GuestAcceptedMemory {
                acceptance_lock: Default::default(),
                bitmap_lock: Default::default(),
                bitmap: GuestMemoryBitmap::new(address_space_size as usize, GuestMemoryBitmapPageSize::PageSize4k)?,
                acceptor: self.acceptor.clone().unwrap(),
            })
        } else {
            None
        };

        // Loop through each of the memory map entries and create a mapping for it.
        for entry in memory_layout.ram() {
            if entry.range.is_empty() {
                continue;
            }
            let base_addr = entry.range.start();
            let file_offset = file_starting_offset.checked_add(base_addr).unwrap();

            tracing::trace!(base_addr, file_offset, "mapping lower ram");

            mapping
                .map_file(
                    base_addr as usize,
                    entry.range.len() as usize,
                    mshv_vtl_low.get(),
                    file_offset,
                    true,
                )
                .map_err(MappingError::Map)?;

            if let Some((bitmaps, state)) = permission_bitmaps
                .as_mut()
                .zip(self.permissions_bitmap_state)
            {
                bitmaps.read_bitmap.init(entry.range, state)?;
                bitmaps.write_bitmap.init(entry.range, state)?;
                bitmaps.kernel_execute_bitmap.init(entry.range, state)?;
                bitmaps.user_execute_bitmap.init(entry.range, state)?;
            }

            if let Some((accepted_memory, state)) = accepted_memory
                .as_mut()
                .zip(self.acceptance_bitmap_state)
            {
                accepted_memory.bitmap.init(entry.range, state)?;
            }

            tracing::trace!(?entry, "mapped memory map entry");
        }

        let registrar = if self.for_kernel_access {
            let mshv = Mshv::new().map_err(MappingError::OpenDevice)?;
            let mshv_vtl = mshv.create_vtl().map_err(MappingError::OpenDevice)?;
            Some(MemoryRegistrar::new(
                memory_layout,
                self.physical_address_base,
                MshvVtlWithPolicy {
                    mshv_vtl,
                    ignore_registration_failure: self.ignore_registration_failure,
                    shared: self.shared,
                },
            ))
        } else {
            None
        };

        Ok(GuestMemoryMapping {
            mapping,
            iova_offset: self.dma_base_address,
            valid_memory: self.valid_memory.clone(),
            permission_bitmaps,
            accepted_memory,
            registrar,
        })
    }
}

impl GuestMemoryMapping {
    /// Create a new builder for a guest memory mapping.
    ///
    /// Map all ranges with a physical address offset of
    /// `physical_address_base`. This can be zero, or the VTOM address for SNP,
    /// or the VTL0 alias address for non-isolated/software-isolated VMs.
    pub fn builder(physical_address_base: u64) -> GuestMemoryMappingBuilder {
        GuestMemoryMappingBuilder {
            physical_address_base,
            valid_memory: None,
            permissions_bitmap_state: None,
            acceptance_bitmap_state: None,
            acceptor: None,
            shared: false,
            for_kernel_access: false,
            dma_base_address: None,
            ignore_registration_failure: false,
        }
    }

    /// Update the permission bitmaps to reflect the given flags.
    /// Panics if the range is outside of guest RAM.
    pub fn update_permission_bitmaps(&self, range: MemoryRange, flags: HvMapGpaFlags) {
        if let Some(bitmaps) = self.permission_bitmaps.as_ref() {
            // TODO GUEST VSM: synchronize with reading the bitmaps
            let _lock = bitmaps.permission_update_lock.lock();
            bitmaps.read_bitmap.update(range, flags.readable());
            bitmaps.write_bitmap.update(range, flags.writable());
            bitmaps
                .kernel_execute_bitmap
                .update(range, flags.kernel_executable());
            bitmaps
                .user_execute_bitmap
                .update(range, flags.user_executable());
        }
    }

    /// Query the permissions for the given gpn.
    /// Panics if the range is outside of guest RAM.
    pub fn query_access_permission(&self, gpn: u64) -> Result<HvMapGpaFlags, VtlPermissionsError> {
        if let Some(bitmaps) = self.permission_bitmaps.as_ref() {
            Ok(HvMapGpaFlags::new()
                .with_readable(bitmaps.read_bitmap.page_state(gpn))
                .with_writable(bitmaps.write_bitmap.page_state(gpn))
                .with_kernel_executable(bitmaps.kernel_execute_bitmap.page_state(gpn))
                .with_user_executable(bitmaps.user_execute_bitmap.page_state(gpn)))
        } else {
            Err(VtlPermissionsError::NoPermissionsTracked)
        }
    }

    /// Zero the given range of memory.
    pub(crate) fn zero_range(
        &self,
        range: MemoryRange,
    ) -> Result<(), sparse_mmap::SparseMappingError> {
        self.mapping
            .fill_at(range.start() as usize, 0, range.len() as usize)
    }

    pub fn accepted_memory(&self) -> Option<&GuestAcceptedMemory> {
        self.accepted_memory.as_ref()
    }
}

/// SAFETY: Implementing the `GuestMemoryAccess` contract, including the
/// size and lifetime of the mappings and bitmaps.
unsafe impl GuestMemoryAccess for GuestMemoryMapping {
    fn mapping(&self) -> Option<NonNull<u8>> {
        NonNull::new(self.mapping.as_ptr().cast())
    }

    fn max_address(&self) -> u64 {
        self.mapping.len() as u64
    }

    fn expose_va(&self, address: u64, len: u64) -> Result<(), GuestMemoryBackingError> {
        if let Some(registrar) = &self.registrar {
            registrar
                .register(address, len)
                .map_err(|start| GuestMemoryBackingError::other(start, RegistrationError))
        } else {
            // TODO: fail this call once we have a way to avoid calling this for
            // user-mode-only accesses to locked memory (e.g., for vmbus ring
            // buffers). We can't fail this for now because TDX cannot register
            // encrypted memory.
            Ok(())
        }
    }

    fn base_iova(&self) -> Option<u64> {
        // When the alias map is configured for this mapping, VTL2-mapped
        // devices need to do DMA with the alias map bit set to avoid DMAing
        // into VTL1 memory.
        self.iova_offset
    }

    fn access_bitmap(&self) -> Option<guestmem::BitmapInfo> {
        // When the permissions bitmaps are available, they take precedence and
        // therefore should be no more permissive than the access bitmap.
        //
        // TODO GUEST VSM: consider being able to dynamically update these
        // bitmaps. There are two scenarios where this would be useful:
        // 1. To reduce memory consumption in cases where the bitmaps aren't
        //    needed, i.e. the guest chooses not to enable guest vsm and VTL 1
        //    gets revoked.
        // 2. Because the related guest memory objects are initialized before
        // VTL 1 is, the code as it currently stands will always enforce vtl 1
        // protections even if VTL 1 hasn't explicitly enabled it. e.g. if VTL 1
        // never enables vtl protections via the vsm partition config, but it
        // still makes hypercalls to modify the vtl protection mask (this is a
        // valid scenario to help set up default protections), these protections
        // will still be enforced. In practice, a well-designed VTL 1 probably
        // would enable vtl protections before allowing VTL 0 to run again, but
        // technically the implementation here is not to spec.
        if let Some(bitmaps) = self.permission_bitmaps.as_ref() {
            Some(guestmem::BitmapInfo {
                read_bitmap: NonNull::new(bitmaps.read_bitmap.as_ptr().cast()).unwrap(),
                write_bitmap: NonNull::new(bitmaps.write_bitmap.as_ptr().cast()).unwrap(),
                bit_offset: 0,
            })
        } else {
            self.valid_memory
                .as_ref()
                .map(|bitmap| bitmap.access_bitmap())
        }
    }

    fn check_guest_page_acceptance(&self, address: u64, len: usize) {

        let Some(accepted_memory) = &self.accepted_memory else {
            return;
        };

        let acceptor = accepted_memory.acceptor.clone();
        let bitmap_page_size = accepted_memory.bitmap_page_size();

        // TODO: GuestMemoryBitmap::init() enforces the alignment by bitmap_page_size for
        // all ranges. Do we still need to verify that gpn_start..gpn_end is within valid
        // range of guest MemoryLayout?

        // TODO: Do we need to validate in mapping to make sure this is all private?
        // Might not be needed because before when we convert a page to shared, we make sure to 
        // first accept that gpa page as large page. So any non-shared porrtion 
        // of that large page should be accepted already.
        let gpn_start = address / bitmap_page_size as u64;
        let gpn_end = (address + len as u64 - 1) / bitmap_page_size as u64 + 1;

        for gpn in gpn_start..gpn_end {
            if accepted_memory.check_acceptance(gpn) {
                // Wait for any pending acceptance operation to complete.
                if accepted_memory.acceptance_lock.try_lock().is_none() {
                    let _lock = accepted_memory.acceptance_lock.lock();
                }
                continue;
            }

            // Accquire the acceptance lock to indicate that an acceptance operation is underway.
            let _lock = accepted_memory.acceptance_lock.lock();

            // Check to see whether this page has already been marked accepted. If not, it must
            // be accepted now.
            if accepted_memory.check_acceptance(gpn) {
               continue;
            }

            let range = MemoryRange::new(gpn * bitmap_page_size as u64..(gpn + 1) * bitmap_page_size as u64);
            accepted_memory.update_acceptance(range, true);
            acceptor.accept_lower_vtl_pages(range)
                .expect("everything should be in a state where we can accept lower pages");
            acceptor.apply_initial_lower_vtl_protections(range).unwrap();
        }
    }
}
