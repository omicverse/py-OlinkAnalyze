#!/usr/bin/env Rscript
# Drive OlinkAnalyze reference on a synthetic NPX TSV dumped by the
# Python side, mirroring tests/_synth.py::generate_synthetic_npx(seed=0).
#
# Usage:
#   Rscript r_reference_driver.R <npx_tsv> <out_dir>
#
# Outputs (in out_dir):
#   ttest.tsv     olink_ttest(variable='Treatment') result
#   wilcox.tsv    olink_wilcox(variable='Treatment') result
#   lmer.tsv      (skipped — synthetic data has no random structure)
#   norm_adj.tsv  bridge normalization Adj_factor (run on a separate pair)
#   timing.tsv    wall-clock per function

suppressPackageStartupMessages({
  library(OlinkAnalyze)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
npx_tsv <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

npx <- read.table(npx_tsv, sep = "\t", header = TRUE,
                  stringsAsFactors = FALSE, check.names = FALSE,
                  na.strings = c("NA",""))

t0 <- proc.time()[[3]]
tt <- olink_ttest(npx, variable = "Treatment")
t1 <- proc.time()[[3]]
ww <- olink_wilcox(npx, variable = "Treatment")
t2 <- proc.time()[[3]]

write.table(as.data.frame(tt), file.path(out_dir, "ttest.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(as.data.frame(ww), file.path(out_dir, "wilcox.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Optional LMM if a Subject column was injected by the caller's wrapper
if ("Subject" %in% colnames(npx)) {
  t3 <- proc.time()[[3]]
  lm <- tryCatch(
    olink_lmer(npx, variable = "Treatment", random = "Subject"),
    error = function(e) { message("olink_lmer failed: ", e$message); NULL }
  )
  t4 <- proc.time()[[3]]
  if (!is.null(lm)) {
    write.table(as.data.frame(lm), file.path(out_dir, "lmer.tsv"),
                sep = "\t", quote = FALSE, row.names = FALSE)
  }
  write.table(data.frame(ttest = t1 - t0, wilcox = t2 - t1, lmer = t4 - t3),
              file.path(out_dir, "timing.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
} else {
  write.table(data.frame(ttest = t1 - t0, wilcox = t2 - t1),
              file.path(out_dir, "timing.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

# v0.2 references — ANOVA / Kruskal-Wallis. The caller passes a 3-group
# 'Group' column (group0/group1/group2) so >2-level tests are valid.
if ("Group" %in% colnames(npx)) {
  av <- tryCatch(
    suppressMessages(olink_anova(npx, variable = "Group", verbose = FALSE)),
    error = function(e) { message("olink_anova failed: ", e$message); NULL }
  )
  if (!is.null(av)) {
    write.table(as.data.frame(av), file.path(out_dir, "anova.tsv"),
                sep = "\t", quote = FALSE, row.names = FALSE)
  }
  kw <- tryCatch(
    suppressMessages(olink_one_non_parametric(npx, variable = "Group",
                                              verbose = FALSE)),
    error = function(e) { message("olink_kruskal failed: ", e$message); NULL }
  )
  if (!is.null(kw)) {
    write.table(as.data.frame(kw), file.path(out_dir, "kruskal.tsv"),
                sep = "\t", quote = FALSE, row.names = FALSE)
  }
}

# Bridge selector reference — run on a separate frame if one is dumped.
bridge_in <- file.path(out_dir, "..", "bridge_npx.tsv")
if (file.exists(bridge_in)) {
  bdf <- read.table(bridge_in, sep = "\t", header = TRUE,
                    stringsAsFactors = FALSE, check.names = FALSE,
                    na.strings = c("NA", ""))
  bs <- tryCatch(
    olink_bridgeselector(bdf, sampleMissingFreq = 0.9, n = 6),
    error = function(e) { message("olink_bridgeselector failed: ",
                                  e$message); NULL }
  )
  if (!is.null(bs)) {
    write.table(as.data.frame(bs), file.path(out_dir, "bridge.tsv"),
                sep = "\t", quote = FALSE, row.names = FALSE)
  }
}
cat("R OlinkAnalyze reference done\n")
