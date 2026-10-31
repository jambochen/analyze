# GAM 平滑項的拆解：M1（近況線性）上分別加 eta、RD、近況（dr 與其差）、局數的平滑項，看哪一組帶來樣本外改善。
library(mgcv)
sp <- commandArgs(trailingOnly = TRUE)[1]
d <- read.csv(sprintf("data/processed/explore_gam_glmm_%s.csv", sp))
tr <- d[d$train == 1, ]
te <- d[d$train == 0, ]
lin <- "prev_draw + prev_win + dr + d_prev_draw + d_prev_win + d_dr + opp_missing"
lin_nodr <- "prev_draw + prev_win + d_prev_draw + d_prev_win + opp_missing"
forms <- list(
  D1_eta = paste("y ~ offset(eta) +", lin, "+ s(eta)"),
  D2_rd = paste("y ~ offset(eta) +", lin, "+ s(rd) + s(opp_rd)"),
  D3_state = paste("y ~ offset(eta) +", lin_nodr, "+ s(dr) + s(d_dr)"),
  D4_games = paste("y ~ offset(eta) +", lin, "+ s(games_in_tour, k = 5)")
)
out <- data.frame(game_id = te$game_id)
for (nm in names(forms)) {
  m <- bam(as.formula(forms[[nm]]), family = ocat(R = 3), data = tr, discrete = TRUE, nthreads = 8)
  p <- predict(m, newdata = te, type = "response", discrete = FALSE, block.size = 20000)
  out[[paste0("ll_", nm)]] <- -log(p[cbind(seq_len(nrow(te)), te$y)])
  cat(nm, "done", "AIC", AIC(m), "\n")
  rm(m, p); gc()
}
write.csv(out, sprintf("data/processed/explore_gam_decomp_loss_%s.csv", sp), row.names = FALSE)
