import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass


# ======================================
# CONFIG – v3 parametreleri
# ======================================

DATA_PATH = "fall25_ie423_project_data_sample.txt"  # kendi full dataset'in neyse onu yaz

# Daha seçici v3
GOAL_THRESHOLD = 4.0    # 3.0 -> 4.0
COOLDOWN = 8            # 5 -> 8
K_STD = 0.5             # CUSUM k = K_STD * sigma
H_STD = 5.0             # 4.0 -> 5.0
MAX_LEAD = 20           # event'ten önceki max kabul edilen lead time
TOP_N_PLOTS = 3         # çizilecek maç sayısı


# ===============================
# Online istatistikler
# ===============================

class StreamStats:
    """
    Welford's online algoritması ile anlık mean / variance.
    Sadece geçmiş gözlemleri kullanır.
    """
    def __init__(self, min_sigma: float = 1e-3):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_sigma = min_sigma

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> float:
        if self.n < 2:
            return self.min_sigma
        var = self.M2 / (self.n - 1)
        sigma = float(np.sqrt(max(var, 0.0)))
        return max(sigma, self.min_sigma)

    def get_mean(self) -> float:
        return self.mean

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0


# ============================
# CUSUM dedektörü
# ============================

class CusumDetector:
    """
    Tek taraflı (yukarı yönlü) CUSUM.
    k_std, h_std parametreleri sigma ile çarpılır.
    """
    def __init__(self, k_std: float = K_STD, h_std: float = H_STD):
        self.k_std = k_std
        self.h_std = h_std
        self.C_pos = 0.0

    def process(self, x: float, mu: float, sigma: float):
        """
        x: gözlem
        mu, sigma: o ana kadarki (geçmiş) tahminler
        return: (signal, C_pos) – signal=1 ise threshold aşılmıştır
        """
        k = self.k_std * sigma
        h = self.h_std * sigma

        # upward CUSUM
        self.C_pos = max(0.0, self.C_pos + (x - mu - k))

        signal = int(self.C_pos > h)
        if signal:
            # alarmdan sonra reset – pattern'i yakaladık varsay
            self.C_pos = 0.0

        return signal, self.C_pos

    def reset(self):
        self.C_pos = 0.0


# ============================
# Veri yükleme & preprocessing
# ============================

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Txt/CSV'yi okur, kolon isimlerini temizler:
      'GOALS - home' -> 'GOALS_home'
    """
    df = pd.read_csv(path)
    df.columns = [
        c.strip().replace(" - ", "_").replace(" ", "_")
        for c in df.columns
    ]
    df = df.sort_values(["match_id", "halftime", "minute"]).reset_index(drop=True)
    return df


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kümülatif kolonlardan diff_* özellikleri üretir.
    diff ilk satırda 0 olsun diye fillna(0) kullanıyoruz.
    """
    df = df.copy()

    cols = [
        "ATTACKS_home", "ATTACKS_away",
        "DANGEROUS_ATTACKS_home", "DANGEROUS_ATTACKS_away",
        "SHOTS_ON_TARGET_home", "SHOTS_ON_TARGET_away",
        "SHOTS_TOTAL_home", "SHOTS_TOTAL_away",
        "CORNERS_home", "CORNERS_away",
        "PENALTIES_home", "PENALTIES_away",
        "YELLOWCARDS_home", "YELLOWCARDS_away",
        "FOULS_home", "FOULS_away",
        "GOALS_home", "GOALS_away",
        "REDCARDS_home", "REDCARDS_away",
    ]

    for col in cols:
        if col in df.columns:
            df[f"diff_{col}"] = (
                df.groupby("match_id")[col]
                  .diff()
                  .fillna(0.0)
            )

    return df


# ==========================
# Event çıkarımı (ground truth)
# ==========================

def extract_events(df: pd.DataFrame):
    """
    GOALS_* ve REDCARDS_* kümülatif kolonlarının diff'lerinden
    event listesi üretir (gol + kırmızı kart).
    return:
      events_by_match: dict[match_id] -> list of dicts
        each: {"minute": int, "team": "home"/"away", "type": "GOAL" or "RED"}
    """
    df = df.sort_values(["match_id", "halftime", "minute"]).copy()
    required = [
        "diff_GOALS_home", "diff_GOALS_away",
        "diff_REDCARDS_home", "diff_REDCARDS_away"
    ]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Column {r} not found – did you call add_diff_features()?")

    events_by_match = defaultdict(list)

    for mid, g in df.groupby("match_id"):
        for _, row in g.iterrows():
            minute = int(row["minute"])

            if row["diff_GOALS_home"] > 0:
                events_by_match[mid].append(
                    {"minute": minute, "team": "home", "type": "GOAL"}
                )
            if row["diff_GOALS_away"] > 0:
                events_by_match[mid].append(
                    {"minute": minute, "team": "away", "type": "GOAL"}
                )
            if row["diff_REDCARDS_home"] > 0:
                events_by_match[mid].append(
                    {"minute": minute, "team": "home", "type": "RED"}
                )
            if row["diff_REDCARDS_away"] > 0:
                events_by_match[mid].append(
                    {"minute": minute, "team": "away", "type": "RED"}
                )

        events_by_match[mid].sort(key=lambda e: e["minute"])

    return events_by_match


# ==========================
# Warning yapısı
# ==========================

@dataclass
class Warning:
    match_id: int
    minute: int
    team: str     # 'home' or 'away'
    score: float  # risk score


# ===========================================
# Rolling feature'lar
# ===========================================

def add_rolling_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Tek maç DataFrame'i g üzerinde rolling window feature'lar ekler.
    5 dakikalık ve 10 dakikalık pencereler.
    """
    g = g.copy()
    # Bu g tek bir match_id olduğu için direkt rolling kullanabiliriz
    for team in ["home", "away"]:
        # 5 ve 10 dakikalık rolling ATTACKS
        if f"diff_ATTACKS_{team}" in g.columns:
            g[f"roll_att_{team}_5"] = (
                g[f"diff_ATTACKS_{team}"]
                .rolling(window=5, min_periods=1)
                .sum()
            )
            g[f"roll_att_{team}_10"] = (
                g[f"diff_ATTACKS_{team}"]
                .rolling(window=10, min_periods=1)
                .sum()
            )

        # 5 ve 10 dakikalık rolling DANGEROUS_ATTACKS
        if f"diff_DANGEROUS_ATTACKS_{team}" in g.columns:
            g[f"roll_da_{team}_5"] = (
                g[f"diff_DANGEROUS_ATTACKS_{team}"]
                .rolling(window=5, min_periods=1)
                .sum()
            )
            g[f"roll_da_{team}_10"] = (
                g[f"diff_DANGEROUS_ATTACKS_{team}"]
                .rolling(window=10, min_periods=1)
                .sum()
            )

        # 5 dakikalık rolling SHOTS_ON_TARGET
        if f"diff_SHOTS_ON_TARGET_{team}" in g.columns:
            g[f"roll_sot_{team}_5"] = (
                g[f"diff_SHOTS_ON_TARGET_{team}"]
                .rolling(window=5, min_periods=1)
                .sum()
            )

        # 5 dakikalık rolling CORNERS
        if f"diff_CORNERS_{team}" in g.columns:
            g[f"roll_cor_{team}_5"] = (
                g[f"diff_CORNERS_{team}"]
                .rolling(window=5, min_periods=1)
                .sum()
            )

    return g


# ===========================================
# Multi-signal risk modeli + CUSUM (match bazında)
# ===========================================

def simulate_match_multisignal(match_df: pd.DataFrame,
                               match_id: int,
                               goal_threshold: float = GOAL_THRESHOLD,
                               cooldown: int = COOLDOWN,
                               k_std: float = K_STD,
                               h_std: float = H_STD):
    """
    Tek bir maç üzerinde:
      - diff_* kolonlarını kullanarak
      - ATTACKS + DANGEROUS_ATTACKS için CUSUM
      - Rolling ATT/DA/SOT/Corners ile multi-signal risk skoru
      - En az 3 bağımsız sinyal + strong signal + threshold koşulu ile Warning üretir.
      - Low-pressure dakikalarda (çok az atak, tehlikeli atak, SOT yok, pen yok) uyarı açmaz.
    """

    g = match_df.sort_values(["halftime", "minute"]).copy()
    g = add_rolling_features(g)

    # CUSUM & stats
    stats_att = {"home": StreamStats(), "away": StreamStats()}
    stats_da = {"home": StreamStats(), "away": StreamStats()}
    cusum_att = {"home": CusumDetector(k_std, h_std),
                 "away": CusumDetector(k_std, h_std)}
    cusum_da = {"home": CusumDetector(k_std, h_std),
                "away": CusumDetector(k_std, h_std)}

    # uyarı arası mesafe
    last_warning = {"home": -1e9, "away": -1e9}

    # Rakip kırmızı kart bilgisi (persistant)
    opponent_red = {"home": False, "away": False}

    # CUSUM grafiği için kolon
    g["CUSUM_ATTACKS_home"] = 0.0
    g["CUSUM_ATTACKS_away"] = 0.0

    warnings = []

    prev_halftime = None

    for idx, row in g.iterrows():
        minute = int(row["minute"])
        halftime = row.get("halftime", None)

        # devre arası reset (istatistik ve CUSUM)
        if prev_halftime is not None and halftime != prev_halftime:
            for t in ["home", "away"]:
                stats_att[t].reset()
                stats_da[t].reset()
                cusum_att[t].reset()
                cusum_da[t].reset()
                last_warning[t] = -1e9
            # opponent_red'i resetlemiyoruz – kırmızı kart kalıcı
        prev_halftime = halftime

        # diff kolon isimleri
        diff_cols = {
            "ATTACKS_home": "diff_ATTACKS_home",
            "ATTACKS_away": "diff_ATTACKS_away",
            "DANGEROUS_ATTACKS_home": "diff_DANGEROUS_ATTACKS_home",
            "DANGEROUS_ATTACKS_away": "diff_DANGEROUS_ATTACKS_away",
            "SHOTS_ON_TARGET_home": "diff_SHOTS_ON_TARGET_home",
            "SHOTS_ON_TARGET_away": "diff_SHOTS_ON_TARGET_away",
            "CORNERS_home": "diff_CORNERS_home",
            "CORNERS_away": "diff_CORNERS_away",
            "PENALTIES_home": "diff_PENALTIES_home",
            "PENALTIES_away": "diff_PENALTIES_away",
            "REDCARDS_home": "diff_REDCARDS_home",
            "REDCARDS_away": "diff_REDCARDS_away",
        }

        # Bu dakikada yeni kırmızı kart var mı? (opponent_red update)
        if diff_cols["REDCARDS_home"] in g.columns and row.get(diff_cols["REDCARDS_home"], 0) > 0:
            opponent_red["away"] = True
        if diff_cols["REDCARDS_away"] in g.columns and row.get(diff_cols["REDCARDS_away"], 0) > 0:
            opponent_red["home"] = True

        # Her iki takım için risk skoru hesapla
        for team in ["home", "away"]:
            opp = "away" if team == "home" else "home"
            score = 0.0
            signal_count = 0  # v3: multi-signal sayacı
            signal_att = False
            signal_da = False

            # isimleri takım bazında seç
            att_col = diff_cols[f"ATTACKS_{team}"]
            da_col = diff_cols[f"DANGEROUS_ATTACKS_{team}"]
            sot_col = diff_cols[f"SHOTS_ON_TARGET_{team}"]
            cor_col = diff_cols[f"CORNERS_{team}"]
            pen_col = diff_cols[f"PENALTIES_{team}"]

            roll_att_5 = g.at[idx, f"roll_att_{team}_5"] if f"roll_att_{team}_5" in g.columns else 0.0
            roll_da_5  = g.at[idx, f"roll_da_{team}_5"]  if f"roll_da_{team}_5"  in g.columns else 0.0
            roll_sot_5 = g.at[idx, f"roll_sot_{team}_5"] if f"roll_sot_{team}_5" in g.columns else 0.0
            roll_cor_5 = g.at[idx, f"roll_cor_{team}_5"] if f"roll_cor_{team}_5" in g.columns else 0.0
            roll_att_10 = g.at[idx, f"roll_att_{team}_10"] if f"roll_att_{team}_10" in g.columns else 0.0
            roll_da_10  = g.at[idx, f"roll_da_{team}_10"]  if f"roll_da_{team}_10"  in g.columns else 0.0

            pens = row.get(pen_col, 0.0) if pen_col in g.columns else 0.0
            sots = row.get(sot_col, 0.0) if sot_col in g.columns else 0.0

            # daha agresif low-pressure filtresi
            low_pressure = (
                roll_att_10 < 5 and      # 10 dk'da 5'ten az atak
                roll_da_10 < 3 and       # 10 dk'da 3'ten az tehlikeli atak
                roll_sot_5 < 1 and       # son 5 dk'da isabetli şut yok
                pens == 0
            )

            # Penaltı (güçlü sinyal)
            if pens > 0:
                score += 6.0
                signal_count += 1

            # Şut isabet (anlık) – >=2 ise sinyal, 1 ise sadece skor katkısı
            if sots >= 2:
                score += 3.5
                signal_count += 1
            elif sots == 1:
                score += 1.0  # sinyal değil, hafif katkı

            # ATTACKS CUSUM
            if att_col in g.columns:
                x_att = float(row.get(att_col, 0.0))
                mu_att = stats_att[team].get_mean()
                sigma_att = stats_att[team].std()
                signal_att, C_att = cusum_att[team].process(x_att, mu_att, sigma_att)
                # CUSUM değerini sakla (grafik için)
                if team == "home":
                    g.at[idx, "CUSUM_ATTACKS_home"] = C_att
                else:
                    g.at[idx, "CUSUM_ATTACKS_away"] = C_att

                if signal_att:
                    score += 2.0
                    signal_count += 1
                # sonra istatistiği güncelle
                stats_att[team].update(x_att)

            # DANGEROUS_ATTACKS CUSUM
            if da_col in g.columns:
                x_da = float(row.get(da_col, 0.0))
                mu_da = stats_da[team].get_mean()
                sigma_da = stats_da[team].std()
                signal_da, _ = cusum_da[team].process(x_da, mu_da, sigma_da)
                if signal_da:
                    score += 2.5
                    signal_count += 1
                stats_da[team].update(x_da)

            # Rolling SOT (5 dk) – en az 2 isabet varsa baskı var diyelim
            if roll_sot_5 >= 2:
                score += 1.0
                signal_count += 1

            # Rolling ATTACKS (5 dk) – 7+ atak
            if roll_att_5 >= 7:
                score += 1.0
                signal_count += 1

            # Rolling DANGEROUS_ATTACKS (5 dk) – 4+ tehlikeli atak
            if roll_da_5 >= 4:
                score += 1.0
                signal_count += 1

            # Kornerler (5 dk içinde 3+ korner)
            if roll_cor_5 >= 3:
                score += 0.5
                signal_count += 1

            # Rakip kırmızı kart varsa tüm skorları biraz boost
            if opponent_red[team]:
                score *= 1.2

            # --- Uyarı üretme kuralı (v3) ---
            if not low_pressure:
                # en az bir "strong" sinyal var mı?
                strong_signal = (
                    pens > 0 or          # penaltı
                    sots >= 2 or         # anlık 2+ isabet
                    signal_att or        # ATT CUSUM sinyali
                    signal_da or         # DA CUSUM sinyali
                    roll_da_5 >= 4       # 5 dk'da 4+ tehlikeli atak
                )

                if (
                    strong_signal and
                    signal_count >= 3 and             # 2→3 yaptık
                    score >= goal_threshold and
                    (minute - last_warning[team]) >= cooldown
                ):
                    warnings.append(
                        Warning(match_id=match_id, minute=minute,
                                team=team, score=score)
                    )
                    last_warning[team] = minute

    return warnings, g


# ======================================
# Uyarıları event'lere eşle ve metrikler
# ======================================

def match_warnings_to_events(warnings, events, max_lead: int = MAX_LEAD):
    """
    warnings: Warning objeleri listesi
    events:   dict {"minute": int, "team": "home"/"away", "type": "GOAL"/"RED"}

    Her event için, aynı takım için max_lead içindeki en geç uyarıyı bul.
    """
    warnings = sorted(warnings, key=lambda w: (w.team, w.minute))
    events = sorted(events, key=lambda e: (e["team"], e["minute"]))

    used_warning_idx = set()
    matched_pairs = []

    for e in events:
        E = e["minute"]
        team = e["team"]

        best_idx = None
        best_W = None
        best_lead = None

        for wi, w in enumerate(warnings):
            if wi in used_warning_idx:
                continue
            if w.team != team:
                continue
            if w.minute >= E:
                continue

            lead = E - w.minute
            if lead <= 0 or lead > max_lead:
                continue

            if best_lead is None or lead < best_lead:
                best_lead = lead
                best_idx = wi
                best_W = w

        if best_idx is not None:
            used_warning_idx.add(best_idx)
            matched_pairs.append(
                {
                    "warning_minute": best_W.minute,
                    "event_minute": E,
                    "team": team,
                    "event_type": e["type"],
                    "lead_time": best_lead,
                    "score": best_W.score,
                }
            )

    true_detections = len(matched_pairs)
    false_alarms = len(warnings) - len(used_warning_idx)
    missed_events = len(events) - true_detections

    return matched_pairs, false_alarms, missed_events


def evaluate_all_matches(df: pd.DataFrame):
    """
    Tüm maçlar için:
      - multi-signal model ile uyarı üret
      - event'leri çıkar
      - per-match metrik + overall metrik hesapla
      - her maçın CUSUM grafiği için df'leri sakla
    """
    df = add_diff_features(df)
    events_by_match = extract_events(df)

    per_match_results = []
    cusum_storage = {}
    all_warnings_global = []
    all_events_global = []

    total_events = 0
    total_warnings = 0
    total_true = 0
    total_false = 0
    total_missed = 0
    all_lead_times = []

    for mid, match_df in df.groupby("match_id"):
        warnings, cusum_df = simulate_match_multisignal(
            match_df, match_id=mid,
            goal_threshold=GOAL_THRESHOLD,
            cooldown=COOLDOWN,
            k_std=K_STD,
            h_std=H_STD
        )
        cusum_storage[mid] = cusum_df

        events = events_by_match.get(mid, [])

        matched_pairs, false_alarms, missed_events = match_warnings_to_events(
            warnings, events, max_lead=MAX_LEAD
        )

        lead_times = [p["lead_time"] for p in matched_pairs]

        result = {
            "match_id": mid,
            "n_events": len(events),
            "n_warnings": len(warnings),
            "true_detections": len(matched_pairs),
            "false_alarms": false_alarms,
            "missed_events": missed_events,
            "lead_times": lead_times,
            "warnings": warnings,
            "events": events,
            "matched_pairs": matched_pairs,
        }
        per_match_results.append(result)

        total_events += len(events)
        total_warnings += len(warnings)
        total_true += len(matched_pairs)
        total_false += false_alarms
        total_missed += missed_events
        all_lead_times.extend(lead_times)

        all_warnings_global.extend(
            [{"match_id": w.match_id, "minute": w.minute, "team": w.team, "score": w.score}
             for w in warnings]
        )
        all_events_global.extend(
            [{"match_id": mid, "minute": e["minute"], "team": e["team"], "type": e["type"]}
             for e in events]
        )

    overall_summary = {
        "total_matches": len(per_match_results),
        "total_events": total_events,
        "total_warnings": total_warnings,
        "total_true_detections": total_true,
        "total_false_alarms": total_false,
        "total_missed_events": total_missed,
        "average_lead_time": float(np.mean(all_lead_times)) if all_lead_times else float("nan"),
        "median_lead_time": float(np.median(all_lead_times)) if all_lead_times else float("nan"),
    }

    warnings_df = pd.DataFrame(all_warnings_global)
    events_df = pd.DataFrame(all_events_global)

    return per_match_results, overall_summary, cusum_storage, warnings_df, events_df


# ===========================
# En iyi maçların CUSUM grafiği
# ===========================

def plot_best_matches(per_match_results, cusum_storage,
                      top_n: int = TOP_N_PLOTS,
                      output_prefix: str = "cusum_multisignal_v3"):
    """
    En iyi lead time'a sahip maçları seçip
    home ATTACKS CUSUM grafiğini çiz.
    """
    candidates = []
    for res in per_match_results:
        if res["lead_times"]:
            min_lead = min(res["lead_times"])
            candidates.append((min_lead, res["match_id"]))

    candidates.sort(key=lambda x: x[0])

    for rank, (lead, mid) in enumerate(candidates[:top_n], start=1):
        df_match = cusum_storage[mid]
        res = next(r for r in per_match_results if r["match_id"] == mid)

        plt.figure(figsize=(10, 5))
        plt.plot(df_match["minute"], df_match["CUSUM_ATTACKS_home"],
                 label="CUSUM (ATTACKS_home)")

        # Uyarılar
        first_warn_labeled = False
        for w in res["warnings"]:
            if w.team == "home":
                label = "Warning (home)" if not first_warn_labeled else None
                first_warn_labeled = True
                plt.axvline(w.minute, linestyle="--", label=label)

        # Event'ler
        first_goal_labeled = False
        first_red_labeled = False
        for e in res["events"]:
            if e["team"] != "home":
                continue
            color = "orange" if e["type"] == "GOAL" else "red"
            if e["type"] == "GOAL":
                label = "GOAL (home)" if not first_goal_labeled else None
                first_goal_labeled = True
            else:
                label = "RED (home)" if not first_red_labeled else None
                first_red_labeled = True
            plt.axvline(e["minute"], color=color, linestyle="-", label=label)

        plt.xlabel("Minute")
        plt.ylabel("CUSUM")
        plt.title(f"Match {mid} - Best lead time {lead} minutes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_match_{mid}.png")
        plt.close()


# ===========================
# MAIN
# ===========================

def main():
    df = load_and_clean_data(DATA_PATH)

    per_match_results, overall_summary, cusum_storage, warnings_df, events_df = evaluate_all_matches(df)

    # Per-match metrikleri yaz
    print("\n===== Per-match metrics (v3) =====")
    for res in per_match_results:
        print(f"\nMatch {res['match_id']}:")
        print(f"  Events         : {res['n_events']}")
        print(f"  Warnings       : {res['n_warnings']}")
        print(f"  True detections: {res['true_detections']}")
        print(f"  False alarms   : {res['false_alarms']}")
        print(f"  Missed events  : {res['missed_events']}")
        print(f"  Lead times     : {res['lead_times']}")
        if res["matched_pairs"]:
            print("  Matched (W -> E, team, type, lead, score):")
            for p in res["matched_pairs"]:
                print(f"    {p['warning_minute']:2d} -> {p['event_minute']:2d}  "
                      f"({p['team']}, {p['event_type']}), "
                      f"lead={p['lead_time']} min, score={p['score']:.2f}")

    # Genel özet
    print("\n===== Overall summary (v3) =====")
    for k, v in overall_summary.items():
        print(f"{k}: {v}")

    # Sonuç CSV'leri
    warnings_df.to_csv("predictions_multisignal_v3.csv", index=False)
    events_df.to_csv("actual_events_from_cumulatives_v3.csv", index=False)
    print("\nSaved: predictions_multisignal_v3.csv, actual_events_from_cumulatives_v3.csv")

    # En iyi maçların CUSUM grafiğini çiz
    plot_best_matches(per_match_results, cusum_storage)


if __name__ == "__main__":
    main()
