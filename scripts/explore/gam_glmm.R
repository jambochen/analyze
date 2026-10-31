# GAM 與 GLMM 的探索：與 state_adjusted_probs.py 同一份樣本與切分，全部用 mgcv 的 ocat 家族，主基準的 eta 作 offset。
# ocat 的切點固定、不隨等級分調整和局帶寬，所以只在 ocat 家族內互相比較，不與正式腳本的絕對對數分數相比。
# M0 只有 offset；M1 近況全放（線性）；M2 在 M1 之上加平滑項；M3 在 M1 之上加玩家隨機截距；M4 兩者都加。
# 逐列損失寫給 gam_glmm_summary.py 算 dyadic 標準誤。
library(mgcv)
args <- commandArgs(trailingOnly = TRUE)
sp <- args[1]
d <- read.csv(sprintf("data/processed/explore_gam_glmm_%s.csv", sp))
tr <- d[d$train == 1, ]
te <- d[d$train == 0, ]
tr$player <- factor(tr$player)
seen <- te$player %in% levels(tr$player)
te$player <- factor(te$player, levels = levels(tr$player))
# 玩家隨機效應的設計矩陣每列有玩家數個欄位，predict 的區塊要小（block.size），否則單一區塊就超過記憶體上限
# NA 的水準會使 predict 丟掉整列；未見過的玩家先指到第一個水準，隨機效應另以 exclude 設為 0
te$player[!seen] <- levels(tr$player)[1]
cat(sp, "train", nrow(tr), "test", nrow(te), "test rows with unseen player", sum(!seen), "\n")

lin <- "prev_draw + prev_win + dr + d_prev_draw + d_prev_win + d_dr + opp_missing"
sm <- "s(eta) + s(rd) + s(opp_rd) + s(games_in_tour, k = 5) + s(dr) + s(d_dr)"
lin_nodr <- "prev_draw + prev_win + d_prev_draw + d_prev_win + opp_missing"
forms <- list(
  M0 = "y ~ offset(eta)",
  M1 = paste("y ~ offset(eta) +", lin),
  M2 = paste("y ~ offset(eta) +", lin_nodr, "+", sm),
  M3 = paste("y ~ offset(eta) +", lin, '+ s(player, bs = "re")'),
  M4 = paste("y ~ offset(eta) +", lin_nodr, "+", sm, '+ s(player, bs = "re")')
)
out <- data.frame(game_id = te$game_id, player = te$player, opponent = te$opponent)
for (nm in names(forms)) {
  t0 <- Sys.time()
  f <- as.formula(forms[[nm]])
  # M0、M1 沒有平滑項，bam 不接受，改用 gam
  m <- if (nm %in% c("M0", "M1")) gam(f, family = ocat(R = 3), data = tr) else bam(f, family = ocat(R = 3), data = tr, discrete = TRUE, nthreads = 8)
  cat(nm, "fit", format(Sys.time() - t0), "AIC", AIC(m), "\n")
  if (nm %in% c("M2", "M4")) print(summary(m)$s.table)
  if (nm %in% c("M3", "M4")) print(gam.vcomp(m))
  pred <- function(nd, ...) if (nm %in% c("M0", "M1")) predict(m, newdata = nd, type = "response", ...) else predict(m, newdata = nd, type = "response", discrete = FALSE, block.size = 2000, ...)
  p <- pred(te)
  # 沒在訓練期出現的玩家，隨機截距取 0
  if (nm %in% c("M3", "M4") && any(!seen)) {
    p[!seen, ] <- pred(te[!seen, ], exclude = "s(player)")
  }
  pr <- p[cbind(seq_len(nrow(te)), te$y)]
  cum <- t(apply(p, 1, cumsum))[, 1:2]
  o <- t(sapply(te$y, function(k) c(k <= 1, k <= 2)))
  out[[paste0("ll_", nm)]] <- -log(pr)
  out[[paste0("rps_", nm)]] <- rowSums((cum - o)^2) / 2
  rm(m, p); gc()
}
write.csv(out, sprintf("data/processed/explore_gam_glmm_loss_%s.csv", sp), row.names = FALSE)
