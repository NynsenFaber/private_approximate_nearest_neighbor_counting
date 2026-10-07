//! Keeps the `README.md` next to each algorithm identical to its rustdoc.
//!
//! The rustdoc comment on each type alias (`pub type Top1 = ...`) is the source.
//! This test renders it as GitHub Markdown and compares it with the README in the
//! same folder: hidden doctest lines (`# ...`) are dropped, code blocks are marked
//! `rust`, headings move down one level, links to the other algorithms point at their READMEs, and the remaining
//! intra-doc links become plain code.
//!
//! The test fails when a README is stale. Regenerate them with
//!
//! ```text
//! UPDATE_READMES=1 cargo test --test algorithm_readmes
//! ```
//!
//! CI does the same on every push to `master` and commits the result.

use std::fs;
use std::path::{Path, PathBuf};

/// (folder, path of the alias in the crate, title of the page).
const PAGES: &[(&str, &str, &str)] = &[
    ("src/anns/top1", "anns::Top1", "Top-1"),
    ("src/anns/close_top1", "anns::CloseTop1", "CloseTop-1"),
    (
        "src/anns/tensor_close_top1",
        "anns::TensorCloseTop1",
        "TensorCloseTop-1",
    ),
    ("src/anns/tensor_top1", "anns::TensorTop1", "TensorTop-1"),
    ("src/annc/top1", "annc::Top1Counter", "Top-1 counting"),
    (
        "src/annc/close_top1",
        "annc::CloseTop1Counter",
        "CloseTop-1 counting",
    ),
    (
        "src/annc/tensor_close_top1",
        "annc::TensorCloseTop1Counter",
        "TensorCloseTop-1 counting",
    ),
    (
        "src/annc/tensor_top1",
        "annc::TensorTop1Counter",
        "TensorTop-1 counting",
    ),
    ("src/annc/dp/top1", "annc::dp::DpTop1", "DPTop-1"),
    (
        "src/annc/dp/close_top1",
        "annc::dp::DpCloseTop1",
        "DP CloseTop-1",
    ),
    (
        "src/annc/dp/tensor_close_top1",
        "annc::dp::DpTensorCloseTop1",
        "DP TensorCloseTop-1",
    ),
    (
        "src/annc/dp/tensor_top1",
        "annc::dp::DpTensorTop1",
        "DP TensorTop-1",
    ),
];

#[test]
fn algorithm_readmes_match_the_rustdoc() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let update = std::env::var_os("UPDATE_READMES").is_some();
    let mut stale = Vec::new();

    for &(folder, alias, title) in PAGES {
        let source = fs::read_to_string(root.join(folder).join("mod.rs")).unwrap();
        let expected = render(folder, alias, title, &doc_comment(&source));
        let readme = root.join(folder).join("README.md");
        if fs::read_to_string(&readme).ok().as_deref() != Some(expected.as_str()) {
            if update {
                fs::write(&readme, &expected).unwrap();
            } else {
                stale.push(readme.display().to_string());
            }
        }
    }

    assert!(
        stale.is_empty(),
        "these READMEs differ from the rustdoc they are generated from:\n  {}\n\
         regenerate them with: UPDATE_READMES=1 cargo test --test algorithm_readmes",
        stale.join("\n  ")
    );
}

/// The `///` lines right above `pub type`, without the `///` prefix.
fn doc_comment(source: &str) -> Vec<String> {
    let lines: Vec<&str> = source.lines().collect();
    let alias = lines
        .iter()
        .position(|line| line.starts_with("pub type "))
        .expect("the module defines a public type alias");
    let mut doc: Vec<String> = lines[..alias]
        .iter()
        .rev()
        .take_while(|line| line.starts_with("///"))
        .map(|line| {
            let text = &line[3..];
            text.strip_prefix(' ').unwrap_or(text).to_string()
        })
        .collect();
    doc.reverse();
    doc
}

fn render(folder: &str, alias: &str, title: &str, doc: &[String]) -> String {
    let mut out = format!(
        "<!-- Generated from the rustdoc of `{folder}/mod.rs` by \
         `UPDATE_READMES=1 cargo test --test algorithm_readmes`. Edit the doc comment, \
         not this file. -->\n\n\
         # {title}\n\n\
         `ann_rust::{alias}` · [all algorithms]({})\n\n",
        relative(folder, "README.md")
    );
    let mut in_code = false;
    let mut rust_code = false;
    for line in doc {
        if let Some(language) = line.trim_start().strip_prefix("```") {
            if in_code {
                out.push_str("```\n");
            } else {
                rust_code = language.is_empty() || language.starts_with("rust");
                out.push_str(if rust_code { "```rust\n" } else { line });
                if !rust_code {
                    out.push('\n');
                }
            }
            in_code = !in_code;
            continue;
        }
        if in_code {
            let hidden = line.trim() == "#" || line.trim_start().starts_with("# ");
            if !(rust_code && hidden) {
                out.push_str(line);
                out.push('\n');
            }
        } else {
            // The page title is the only level-1 heading.
            if line.starts_with('#') {
                out.push('#');
            }
            out.push_str(&convert_links(line, folder));
            out.push('\n');
        }
    }
    out
}

/// Rewrites the intra-doc links of one line of prose.
fn convert_links(line: &str, folder: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            // Inline code is copied untouched, brackets included.
            '`' => {
                let end = find(&chars, i + 1, '`').unwrap_or(chars.len() - 1);
                out.extend(&chars[i..=end]);
                i = end + 1;
            }
            '[' => {
                let Some(close) = find(&chars, i + 1, ']') else {
                    out.push('[');
                    i += 1;
                    continue;
                };
                let text: String = chars[i + 1..close].iter().collect();
                let target = if chars.get(close + 1) == Some(&'(') {
                    find(&chars, close + 2, ')')
                        .map(|end| (chars[close + 2..end].iter().collect::<String>(), end))
                } else {
                    None
                };
                match target {
                    Some((target, end)) => {
                        out.push_str(&link(&text, &target, folder));
                        i = end + 1;
                    }
                    // `[`path`]`: an intra-doc link whose text is its target.
                    None if text.starts_with('`') && text.ends_with('`') => {
                        let path = text.trim_matches('`');
                        out.push_str(&link(&format!("`{}`", short(path)), path, folder));
                        i = close + 1;
                    }
                    None => {
                        out.push('[');
                        i += 1;
                    }
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// A link to another algorithm's README, an external URL kept as is, or plain text.
fn link(text: &str, target: &str, folder: &str) -> String {
    if target.contains("://") {
        return format!("[{text}]({target})");
    }
    let layer = PAGES
        .iter()
        .find(|page| page.0 == folder)
        .map(|page| page.1.rsplit_once("::").unwrap().0)
        .unwrap();
    let path = match target.strip_prefix("super::") {
        Some(rest) => format!("{layer}::{rest}"),
        None => short(target).to_string(),
    };
    match PAGES.iter().find(|page| page.1 == path) {
        Some(page) => format!(
            "[{text}]({})",
            relative(folder, &format!("{}/README.md", page.0))
        ),
        None => text.to_string(),
    }
}

fn short(path: &str) -> &str {
    path.strip_prefix("crate::").unwrap_or(path)
}

fn find(chars: &[char], from: usize, wanted: char) -> Option<usize> {
    (from..chars.len()).find(|&j| chars[j] == wanted)
}

/// Path of `to` (relative to the repository root) seen from the folder `from`.
fn relative(from: &str, to: &str) -> String {
    let from: Vec<&str> = from.split('/').collect();
    let to: Vec<&str> = to.split('/').collect();
    let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    let mut path = PathBuf::new();
    for _ in common..from.len() {
        path.push("..");
    }
    for part in &to[common..] {
        path.push(part);
    }
    path.to_string_lossy().into_owned()
}
