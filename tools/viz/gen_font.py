"""Generate the bitmap font of the PNG renderer.

The glyphs below (five columns; seven rows down to the line that the letters stand on,
and two more below it for the letters that reach below; printable ASCII in order from the
space to the tilde) were drawn for mopt-rs. A glyph that is given with seven rows has
nothing below the line. Run from the repository root:

    python -I tools/viz/gen_font.py .

and the table in ``crates/mopt-rs/src/core_infra/viz/font.rs`` is written anew.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import sys
from pathlib import Path

FIRST = 0x20
LAST = 0x7E
COLUMNS = 5
ROWS = 9
ROWS_ABOVE = 7

GLYPHS = r"""
.....
.....
.....
.....
.....
.....
.....

..#..
..#..
..#..
..#..
..#..
.....
..#..

.#.#.
.#.#.
.#.#.
.....
.....
.....
.....

.#.#.
.#.#.
#####
.#.#.
#####
.#.#.
.#.#.

..#..
.####
#.#..
.###.
..#.#
####.
..#..

##...
##..#
...#.
..#..
.#...
#..##
...##

.##..
#..#.
#.#..
.#...
#.#.#
#..#.
.##.#

..#..
..#..
..#..
.....
.....
.....
.....

...#.
..#..
.#...
.#...
.#...
..#..
...#.

.#...
..#..
...#.
...#.
...#.
..#..
.#...

.....
..#..
#.#.#
.###.
#.#.#
..#..
.....

.....
..#..
..#..
#####
..#..
..#..
.....

.....
.....
.....
.....
.##..
..#..
.#...

.....
.....
.....
#####
.....
.....
.....

.....
.....
.....
.....
.....
.##..
.##..

.....
....#
...#.
..#..
.#...
#....
.....

.###.
#...#
#..##
#.#.#
##..#
#...#
.###.

..#..
.##..
..#..
..#..
..#..
..#..
.###.

.###.
#...#
....#
...#.
..#..
.#...
#####

#####
...#.
..#..
...#.
....#
#...#
.###.

...#.
..##.
.#.#.
#..#.
#####
...#.
...#.

#####
#....
####.
....#
....#
#...#
.###.

..##.
.#...
#....
####.
#...#
#...#
.###.

#####
....#
...#.
..#..
.#...
.#...
.#...

.###.
#...#
#...#
.###.
#...#
#...#
.###.

.###.
#...#
#...#
.####
....#
...#.
.##..

.....
.##..
.##..
.....
.##..
.##..
.....

.....
.##..
.##..
.....
.##..
..#..
.#...

...#.
..#..
.#...
#....
.#...
..#..
...#.

.....
.....
#####
.....
#####
.....
.....

.#...
..#..
...#.
....#
...#.
..#..
.#...

.###.
#...#
....#
...#.
..#..
.....
..#..

.###.
#...#
#.###
#.#.#
#.###
#....
.###.

.###.
#...#
#...#
#...#
#####
#...#
#...#

####.
#...#
#...#
####.
#...#
#...#
####.

.###.
#...#
#....
#....
#....
#...#
.###.

###..
#..#.
#...#
#...#
#...#
#..#.
###..

#####
#....
#....
####.
#....
#....
#####

#####
#....
#....
####.
#....
#....
#....

.###.
#...#
#....
#.###
#...#
#...#
.####

#...#
#...#
#...#
#####
#...#
#...#
#...#

.###.
..#..
..#..
..#..
..#..
..#..
.###.

..###
...#.
...#.
...#.
...#.
#..#.
.##..

#...#
#..#.
#.#..
##...
#.#..
#..#.
#...#

#....
#....
#....
#....
#....
#....
#####

#...#
##.##
#.#.#
#.#.#
#...#
#...#
#...#

#...#
#...#
##..#
#.#.#
#..##
#...#
#...#

.###.
#...#
#...#
#...#
#...#
#...#
.###.

####.
#...#
#...#
####.
#....
#....
#....

.###.
#...#
#...#
#...#
#.#.#
#..#.
.##.#

####.
#...#
#...#
####.
#.#..
#..#.
#...#

.####
#....
#....
.###.
....#
....#
####.

#####
..#..
..#..
..#..
..#..
..#..
..#..

#...#
#...#
#...#
#...#
#...#
#...#
.###.

#...#
#...#
#...#
#...#
#...#
.#.#.
..#..

#...#
#...#
#...#
#.#.#
#.#.#
#.#.#
.#.#.

#...#
#...#
.#.#.
..#..
.#.#.
#...#
#...#

#...#
#...#
#...#
.#.#.
..#..
..#..
..#..

#####
....#
...#.
..#..
.#...
#....
#####

.###.
.#...
.#...
.#...
.#...
.#...
.###.

.....
#....
.#...
..#..
...#.
....#
.....

.###.
...#.
...#.
...#.
...#.
...#.
.###.

..#..
.#.#.
#...#
.....
.....
.....
.....

.....
.....
.....
.....
.....
.....
#####

.#...
..#..
...#.
.....
.....
.....
.....

.....
.....
.###.
....#
.####
#...#
.####

#....
#....
#.##.
##..#
#...#
#...#
####.

.....
.....
.###.
#....
#....
#...#
.###.

....#
....#
.##.#
#..##
#...#
#...#
.####

.....
.....
.###.
#...#
#####
#....
.###.

..##.
.#..#
.#...
###..
.#...
.#...
.#...

.....
.....
.####
#...#
#...#
#...#
.####
....#
.###.

#....
#....
#.##.
##..#
#...#
#...#
#...#

..#..
.....
.##..
..#..
..#..
..#..
.###.

...#.
.....
..##.
...#.
...#.
...#.
...#.
#..#.
.##..

#....
#....
#..#.
#.#..
##...
#.#..
#..#.

.##..
..#..
..#..
..#..
..#..
..#..
.###.

.....
.....
##.#.
#.#.#
#.#.#
#...#
#...#

.....
.....
#.##.
##..#
#...#
#...#
#...#

.....
.....
.###.
#...#
#...#
#...#
.###.

.....
.....
####.
#...#
#...#
#...#
####.
#....
#....

.....
.....
.####
#...#
#...#
#...#
.####
....#
....#

.....
.....
#.##.
##..#
#....
#....
#....

.....
.....
.####
#....
.###.
....#
####.

.#...
.#...
###..
.#...
.#...
.#..#
..##.

.....
.....
#...#
#...#
#...#
#..##
.##.#

.....
.....
#...#
#...#
#...#
.#.#.
..#..

.....
.....
#...#
#...#
#.#.#
#.#.#
.#.#.

.....
.....
#...#
.#.#.
..#..
.#.#.
#...#

.....
.....
#...#
#...#
#...#
#...#
.####
....#
.###.

.....
.....
#####
...#.
..#..
.#...
#####

...#.
..#..
..#..
.#...
..#..
..#..
...#.

..#..
..#..
..#..
..#..
..#..
..#..
..#..

.#...
..#..
..#..
...#.
..#..
..#..
.#...

.....
.....
.#...
#.#.#
...#.
.....
.....
"""

HEADER = """//! The bitmap font of the PNG renderer: printable ASCII, five columns by nine rows (seven
//! down to the line that the letters stand on, two below it).
//!
//! Generated by `tools/viz/gen_font.py`, where the glyphs are drawn; do not edit the table
//! by hand.
//!
//! MultiOptPy: no counterpart (new in mopt-rs) (MultiOptPy draws through matplotlib).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

/// Columns of a glyph.
pub const GLYPH_COLUMNS: usize = 5;
/// Rows of a glyph.
pub const GLYPH_ROWS: usize = 9;
/// Rows of a glyph down to the line that the letters stand on (the height of a capital).
pub const GLYPH_ROWS_ABOVE: usize = 7;
/// Columns from one character to the next (a glyph and one empty column).
pub const GLYPH_ADVANCE: usize = GLYPH_COLUMNS + 1;

const FIRST: u32 = 0x20;
const LAST: u32 = 0x7e;

/// The rows of the glyph of `c` from the top, the leftmost column in bit 4. Characters
/// outside printable ASCII are drawn as a question mark.
pub fn glyph(c: char) -> [u8; GLYPH_ROWS] {
    let code = u32::from(c);
    let code = if (FIRST..=LAST).contains(&code) {
        code
    } else {
        u32::from('?')
    };
    GLYPHS[(code - FIRST) as usize]
}

/// Whether the pixel in `column` and `row` of the glyph of `c` is set.
pub fn glyph_pixel(c: char, column: usize, row: usize) -> bool {
    column < GLYPH_COLUMNS
        && row < GLYPH_ROWS
        && glyph(c)[row] & (1 << (GLYPH_COLUMNS - 1 - column)) != 0
}

#[rustfmt::skip]
const GLYPHS: [[u8; GLYPH_ROWS]; (LAST - FIRST + 1) as usize] = [
"""

FOOTER = """];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyphs_are_found_and_distinct() {
        // The capital T: a bar on top of a stem.
        assert_eq!(
            glyph('T'),
            [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x00]
        );
        assert!(glyph_pixel('T', 0, 0) && glyph_pixel('T', 2, 6));
        assert!(!glyph_pixel('T', 0, 6) && !glyph_pixel('T', 5, 0) && !glyph_pixel('T', 2, 7));
        // Only letters like g reach below the line.
        assert!(glyph_pixel('g', 1, 8) && !glyph_pixel('g', 0, 9));
        assert!(glyph('a')[GLYPH_ROWS_ABOVE..].iter().all(|row| *row == 0));
        assert_eq!(glyph(' '), [0; GLYPH_ROWS]);
        // No two printable characters look the same, and only the space is empty.
        let all: Vec<char> = (FIRST..=LAST).filter_map(char::from_u32).collect();
        for (i, a) in all.iter().enumerate() {
            assert!(*a == ' ' || glyph(*a) != [0; GLYPH_ROWS], "{a}");
            for b in &all[i + 1..] {
                assert_ne!(glyph(*a), glyph(*b), "{a} and {b}");
            }
        }
        // Anything else is a question mark.
        assert_eq!(glyph('\\u{e5}'), glyph('?'));
    }
}
"""


def rows_of(art: str) -> list[list[int]]:
    """The rows of every glyph as numbers, the leftmost column in the highest bit."""
    glyphs = [block.split("\n") for block in art.strip("\n").split("\n\n")]
    if len(glyphs) != LAST - FIRST + 1:
        raise SystemExit(f"{len(glyphs)} glyphs, expected {LAST - FIRST + 1}")
    out = []
    for number, glyph in enumerate(glyphs):
        name = chr(FIRST + number)
        if len(glyph) == ROWS_ABOVE:
            glyph = glyph + ["." * COLUMNS] * (ROWS - ROWS_ABOVE)
        if len(glyph) != ROWS or any(len(row) != COLUMNS for row in glyph):
            raise SystemExit(f"glyph {name!r} is not {COLUMNS} by {ROWS_ABOVE} or {ROWS}")
        if any(set(row) - {"#", "."} for row in glyph):
            raise SystemExit(f"glyph {name!r} has other characters than '#' and '.'")
        out.append([int(row.replace("#", "1").replace(".", "0"), 2) for row in glyph])
    return out


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    lines = []
    for number, rows in enumerate(rows_of(GLYPHS)):
        name = chr(FIRST + number)
        shown = "space" if name == " " else name
        values = ", ".join(f"0x{row:02x}" for row in rows)
        lines.append(f"    [{values}], // {shown}\n")
    target = root / "crates" / "mopt-rs" / "src" / "core_infra" / "viz" / "font.rs"
    target.write_text(HEADER + "".join(lines) + FOOTER, encoding="utf-8", newline="\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
