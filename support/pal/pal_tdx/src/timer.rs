// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! TDX Timer implementation for pal_async.

use pal_async::timer::{Instant, PollTimer};
use std::task::{Context, Poll};

/// A TSC deadline timer implementation for TDX
#[derive(Debug)]
pub struct Timer {
    target_deadline: Instant,
    submitted: bool,
}

/// A TSC deadline timer implementation for TDX
impl Timer {
    /// Creates a new instance for the TDX timer.
    pub fn new() -> Self {
        Self {
            target_deadline: Instant::from_nanos(0),
            submitted: false,
        }
    }
}

impl PollTimer for Timer {
    fn poll_timer(&mut self, cx: &mut Context<'_>, deadline: Option<Instant>) -> Poll<Instant> {
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
