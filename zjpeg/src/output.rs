use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Resolved output configuration from CLI flags.
pub struct OutputConfig {
    /// Target directory (if -o points to a dir).
    pub target_dir: Option<PathBuf>,
    /// Target file (if -o points to a single file).
    pub target_file: Option<PathBuf>,
    /// In-place overwrite mode.
    pub in_place: bool,
    /// Suffix to append before extension (e.g. ".optimized").
    pub suffix: String,
    /// Allow overwriting existing files.
    pub force: bool,
    /// Don't write anything.
    pub dry_run: bool,
}

impl OutputConfig {
    /// Create output config from CLI arguments.
    pub fn new(
        output: Option<PathBuf>,
        in_place: bool,
        suffix: String,
        force: bool,
        dry_run: bool,
    ) -> Result<Self> {
        if in_place && !force {
            bail!("--in-place requires --force to prevent accidental overwrites");
        }

        let (target_dir, target_file) = match output {
            Some(ref p) if p.is_dir() || p.to_string_lossy().ends_with('/') => {
                (Some(p.clone()), None)
            }
            Some(p) => (None, Some(p)),
            None => (None, None),
        };

        Ok(Self {
            target_dir,
            target_file,
            in_place,
            suffix,
            force,
            dry_run,
        })
    }

    /// Resolve the output path for a given input file.
    ///
    /// Rules:
    /// - `--in-place` → same path as input
    /// - `-o file.jpg` → that exact path (single-input only)
    /// - `-o dir/` → dir/original_name.jpg
    /// - default → input_stem.{suffix}.ext alongside input
    pub fn resolve(&self, input: &Path, is_single_file: bool) -> Result<PathBuf> {
        if self.in_place {
            return Ok(input.to_path_buf());
        }

        if let Some(ref target_file) = self.target_file {
            if !is_single_file {
                bail!(
                    "cannot use -o with a single file path for multiple inputs; use -o dir/ instead"
                );
            }
            return Ok(target_file.clone());
        }

        if let Some(ref target_dir) = self.target_dir {
            let filename = input.file_name().context("input has no filename")?;
            return Ok(target_dir.join(filename));
        }

        // Default: add suffix before extension
        let stem = input
            .file_stem()
            .context("input has no stem")?
            .to_string_lossy();
        let ext = input
            .extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_else(|| "jpg".into());
        let parent = input.parent().unwrap_or(Path::new("."));

        let new_name = if self.suffix.is_empty() {
            format!("{stem}.{ext}")
        } else {
            format!("{stem}{}.{ext}", self.suffix)
        };

        Ok(parent.join(new_name))
    }

    /// Check whether an output path is safe to write.
    pub fn check_writable(&self, output: &Path, input: &Path) -> Result<()> {
        if self.dry_run {
            return Ok(());
        }

        // Same path as input without --in-place
        if output == input && !self.in_place {
            bail!(
                "output would overwrite input '{}'; use --in-place --force to overwrite",
                input.display()
            );
        }

        // Output exists without --force
        if output.exists() && !self.force && !self.in_place {
            bail!(
                "output '{}' already exists; use --force to overwrite",
                output.display()
            );
        }

        Ok(())
    }

    /// Ensure the parent directory exists.
    pub fn ensure_parent(path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create directory '{}'", parent.display())
                })?;
            }
        }
        Ok(())
    }
}
