//! Minimal `--key value` command line parsing for the experiment binaries.
//!
//! Deliberately dependency free. Unknown options are rejected so that a typo in a
//! long experiment invocation fails immediately instead of silently using a default.

use std::collections::HashMap;

/// Parsed command line arguments.
pub struct Args {
    values: HashMap<String, String>,
}

impl Args {
    /// Parses `std::env::args`, accepting only the options in `known`.
    ///
    /// Supported forms: `--key value`, `--key=value` and the boolean flag `--key`.
    pub fn parse(known: &[&str]) -> Result<Self, String> {
        Self::from_iter(std::env::args().skip(1), known)
    }

    /// Same as [`Args::parse`] but over an explicit argument list (used in tests).
    pub fn from_iter<I: IntoIterator<Item = String>>(
        arguments: I,
        known: &[&str],
    ) -> Result<Self, String> {
        let mut values = HashMap::new();
        let mut iterator = arguments.into_iter().peekable();
        while let Some(argument) = iterator.next() {
            let Some(stripped) = argument.strip_prefix("--") else {
                return Err(format!("unexpected argument '{argument}', options start with --"));
            };
            let (key, value) = match stripped.split_once('=') {
                Some((key, value)) => (key.to_string(), value.to_string()),
                None => {
                    let takes_value = iterator
                        .peek()
                        .map(|next| !next.starts_with("--"))
                        .unwrap_or(false);
                    if takes_value {
                        (stripped.to_string(), iterator.next().unwrap())
                    } else {
                        (stripped.to_string(), "true".to_string())
                    }
                }
            };
            if !known.contains(&key.as_str()) {
                return Err(format!(
                    "unknown option '--{key}'; known options: {}",
                    known.join(", ")
                ));
            }
            values.insert(key, value);
        }
        Ok(Args { values })
    }

    /// `true` if the option was given at all.
    pub fn has(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Reads a value of any parsable type, falling back to `default`.
    pub fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.values.get(key) {
            None => Ok(default),
            Some(raw) => raw
                .parse::<T>()
                .map_err(|_| format!("invalid value '{raw}' for --{key}")),
        }
    }

    /// Reads an optional value.
    pub fn get_optional<T: std::str::FromStr>(&self, key: &str) -> Result<Option<T>, String> {
        match self.values.get(key) {
            None => Ok(None),
            Some(raw) => raw
                .parse::<T>()
                .map(Some)
                .map_err(|_| format!("invalid value '{raw}' for --{key}")),
        }
    }

    /// Reads a boolean flag (`--flag`, `--flag true`, `--flag false`).
    pub fn flag(&self, key: &str, default: bool) -> Result<bool, String> {
        self.get::<bool>(key, default)
    }

    /// Reads a comma separated list, e.g. `--epsilons 0.5,1,2`.
    pub fn get_list<T: std::str::FromStr>(&self, key: &str, default: Vec<T>) -> Result<Vec<T>, String> {
        match self.values.get(key) {
            None => Ok(default),
            Some(raw) => raw
                .split(',')
                .map(|item| {
                    item.trim()
                        .parse::<T>()
                        .map_err(|_| format!("invalid value '{item}' in --{key}"))
                })
                .collect(),
        }
    }

    /// Reads a string option.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|value| value.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(raw: &[&str], known: &[&str]) -> Result<Args, String> {
        Args::from_iter(raw.iter().map(|item| item.to_string()), known)
    }

    #[test]
    fn test_parsing_forms() {
        let parsed = args(
            &["--n", "1000", "--alpha=0.9", "--strict", "--epsilons", "0.5,1,2"],
            &["n", "alpha", "strict", "epsilons"],
        )
        .unwrap();
        assert_eq!(parsed.get::<usize>("n", 0).unwrap(), 1000);
        assert_eq!(parsed.get::<f64>("alpha", 0.).unwrap(), 0.9);
        assert!(parsed.flag("strict", false).unwrap());
        assert_eq!(
            parsed.get_list::<f64>("epsilons", vec![]).unwrap(),
            vec![0.5, 1.0, 2.0]
        );
        assert_eq!(parsed.get::<usize>("missing", 7).unwrap(), 7);
        assert!(!parsed.has("missing"));
    }

    #[test]
    fn test_errors() {
        assert!(args(&["--typo", "1"], &["n"]).is_err());
        assert!(args(&["n", "1"], &["n"]).is_err());
        assert!(args(&["--n", "abc"], &["n"]).unwrap().get::<usize>("n", 0).is_err());
    }

    #[test]
    fn test_optional() {
        let parsed = args(&["--t", "5"], &["t", "m-sub"]).unwrap();
        assert_eq!(parsed.get_optional::<usize>("t").unwrap(), Some(5));
        assert_eq!(parsed.get_optional::<usize>("m-sub").unwrap(), None);
    }
}
