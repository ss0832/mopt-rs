//! Calendar time for run-directory names (UTC; the standard library has no time-zone data).
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy uses `datetime` for run directory names
//!  in `optimization.py:Optimize._setup_directory`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::time::{SystemTime, UNIX_EPOCH};

/// Broken-down UTC time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateTime {
    /// Year.
    pub year: i64,
    /// Month (1-12).
    pub month: u32,
    /// Day (1-31).
    pub day: u32,
    /// Hour (0-23).
    pub hour: u32,
    /// Minute.
    pub minute: u32,
    /// Second.
    pub second: u32,
    /// Microsecond.
    pub micro: u32,
}

/// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

impl DateTime {
    /// Time from microseconds since the Unix epoch.
    pub fn from_unix_micros(us: i64) -> Self {
        let secs = us.div_euclid(1_000_000);
        let micro = us.rem_euclid(1_000_000) as u32;
        let days = secs.div_euclid(86_400);
        let sod = secs.rem_euclid(86_400);
        let (year, month, day) = civil_from_days(days);
        Self {
            year,
            month,
            day,
            hour: (sod / 3600) as u32,
            minute: ((sod % 3600) / 60) as u32,
            second: (sod % 60) as u32,
            micro,
        }
    }

    /// Current UTC time.
    pub fn now_utc() -> Self {
        let us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX));
        Self::from_unix_micros(us)
    }

    /// `YYYY_MM_DD` (MultiOptPy's date directory).
    pub fn date_dir(&self) -> String {
        format!("{:04}_{:02}_{:02}", self.year, self.month, self.day)
    }

    /// `YYYY_MM_DD_HH_MM_SS_ffff` (MultiOptPy's timestamp: microseconds cut to four digits).
    pub fn timestamp(&self) -> String {
        format!(
            "{}_{:02}_{:02}_{:02}_{:04}",
            self.date_dir(),
            self.hour,
            self.minute,
            self.second,
            self.micro / 100
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_dates() {
        let t = DateTime::from_unix_micros(0);
        assert_eq!(t.timestamp(), "1970_01_01_00_00_00_0000");
        // 2026-11-09 12:34:56.789012 UTC.
        let t = DateTime::from_unix_micros(1_794_227_696_789_012);
        assert_eq!(t.timestamp(), "2026_11_09_12_34_56_7890");
        // Leap day.
        let t = DateTime::from_unix_micros(951_782_400_000_000);
        assert_eq!(t.date_dir(), "2000_02_29");
    }
}
