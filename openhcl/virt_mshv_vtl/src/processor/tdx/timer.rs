// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! TDX Timer implementation for pal_async.
/*
use pal_async::timer::{Instant, PollTimer};
use std::task::{Context, Poll};
use hcl::ioctl::ProcessorRunner;
use hcl::ioctl::tdx::Tdx;


/// A TSC deadline timer implementation based on TDX L2 TSC deadline service
//#[derive(Debug)]
pub struct DeadlineTimer<'a> {
    runner: &'a ProcessorRunner<'a, Tdx<'a>>,
    target_deadline: Instant,
    submitted: bool,
}

/// A TSC deadline timer implementation for TDX
impl<'a> DeadlineTimer<'a> {
    /// Creates a new instance for the TDX timer.
    pub fn new(runner: &'a ProcessorRunner<'a, Tdx<'a>>) -> Self {
        Self {
            runner,
            target_deadline: Instant::from_nanos(0),
            submitted: false,
        }
    }
}

impl<'a> PollTimer for DeadlineTimer<'a> {
    fn poll_timer(&mut self, _cx: &mut Context<'_>, deadline: Option<Instant>) -> Poll<Instant> {
        if let Some(deadline) = deadline {
            self.set_deadline(deadline);
        }
    loop {
        let now = Instant::now();
        if self.target_deadline <= now {
            break Poll::Ready(now);
        } else if self.submitted {
            // If the timer was already submitted, we need to wait for it to complete.
            // TODO: Save cx so it can be called from TDX run_vp()
            self.submitted = false;
            break Poll::Pending;
        } else {

            // TODO: Issue TDG.VP.WR
            self.submitted = true;
        }
    }     
    }

    fn set_deadline(&mut self, deadline: Instant) {
        if self.submitted {
            if self.target_deadline > deadline {
                // TODO: Issue TDG.VP.WR 

                // Open: Can this cause problem with poll_timer returning self.submitted = false
            }
        }
        self.target_deadline = deadline;
    }
}

*/