# Practical-significance thresholds (SESOI): basis, caveats and sources

The thresholds live in `SESOI` in `analysis/stage4_report.py`. They were approved provisionally on 2026-11-05, before
the final models were fit. **The basis and caveats below were added on 2026-11-09, after the results were seen. No
value was changed, and the decision not to change pH is recorded below.**

## What they are, and what they are not

A practical-significance threshold (*smallest effect size of interest*, SESOI) is the smallest change this analysis
treats as worth distinguishing from no change. It is used only to classify estimates:

- **detected:** the 95% CI excludes 0. This verdict **doesn't depend on the threshold**.
- **no meaningful change:** the whole 95% CI lies inside ± threshold.
- **inconclusive:** neither of the above.

The thresholds also scale the "effect ÷ threshold" charts and the power analysis (MDE ÷ threshold and power at the
threshold). Changing a threshold needs no refit: change `SESOI` and rerun the report.

Statement for the methods section:

> Practical-significance thresholds were selected by the analyst to distinguish statistically detectable changes from
> changes considered potentially meaningful for environmental interpretation. They are provisional analytical
> thresholds, not regulatory standards or established ecological thresholds. Their selection was informed by published
> research on aquatic-life responses, water-quality criteria and measurement uncertainty. Because ecological responses
> depend on baseline conditions, exposure duration, species and region, the thresholds are screening criteria for a
> national analysis, not measures of ecological harm or benefit.

Three kinds of evidence bear on them, and none fixes a value by itself:

1. **Ecological relevance:** whether the measure matters biologically. Published criteria are almost all
   *absolute* concentrations (e.g. dissolved oxygen < 5 mg/L, conductivity 300 µS/cm, ecoregional phosphorus), not
   sizes of *change*. So they support why a measure matters, not the size of its threshold.
2. **Measurement uncertainty:** whether a change of that size exceeds field-instrument accuracy (USGS, 2023).
3. **Analyst judgment:** what size of change is worth separating from none in a national screening analysis.

## The thresholds

The log-scale measures use log10 units in the model, so ±25% means log10(1.25). On that scale a decrease of the same
size is smaller in percent: +25% matches −20%.

| Measure | Threshold | Instrument accuracy (USGS, 2023, table 6.8–2) | Literature support for this *value* | Main caveat |
|---|---|---|---|---|
| Water temperature | ±0.5 °C | ±0.2 °C | moderate | Biological effects depend on maxima, duration and degree-days, not just a change in the mean (Heck et al., 2018). |
| Dissolved oxygen | ±0.5 mg/L | ±0.1 mg/L (optical) | moderate–high | Importance depends on the starting level: 0.5 mg/L matters more near 5 mg/L, the level EPA describes as stressful for fish, than at saturation (U.S. EPA, n.d.). Impoundments in Massachusetts lowered DO by about 1.15 mg/L on average, and DO recovered within a year at 80% of removal sites (Abbott et al., 2022). |
| pH | ±0.1 units | ±0.1 pH units | low | The threshold equals instrument accuracy (see below). The EPA aquatic-life range is 6.5–9 (U.S. EPA, 1986), an absolute range. |
| Specific conductance | ±10% | ±5% of reading or ±2 µS/cm | moderate | The ecological benchmark (300 µS/cm, central Appalachia; U.S. EPA, 2011) is absolute and regional. The effect depends on the baseline and on which ions are present. |
| Total dissolved solids | ±10% | (laboratory) | low | The 500 mg/L EPA value is a secondary drinking-water standard (aesthetic), not an aquatic-life criterion. TDS is not interchangeable with conductance. |
| Total phosphorus | ±20% | (laboratory) | moderate | Nutrient criteria are ecoregional and absolute (U.S. EPA, 2000; Chambers et al., 2012). A % change means different things at different baselines. |
| Orthophosphate | ±20% | (laboratory) | low | More bioavailable than total P, and no directly applicable benchmark exists. EPA's ecoregional criteria are for total P and shouldn't be transferred. |
| Turbidity | ±25% | stabilization ±0.5 TU or 5% (≤ 100 TU) | moderate–low | An optical, indirect indicator of sediment. The particle-to-turbidity relationship varies, and turbidity units differ between instruments. |
| SSC | ±25% | (laboratory) | moderate–low | Biological effects depend on concentration, duration, particle size and species. No universal % change is established. |
| TSS | ±25% | (laboratory) | low | TSS and SSC are not comparable. TSS is unreliable for natural water and tends to read lower than SSC when sand is present (Gray et al., 2000). The same % is not the same physical change in both. |

"Literature support" rates how well published work supports the specific number, not whether the measure matters
ecologically.

## pH: kept at ±0.1 (decided 2026-11-09)

The literature review flagged pH as the weakest-supported threshold: ±0.1 equals the stated accuracy of common pH
sensors (USGS, 2023), and no ecological source gives a size of pH change that matters. ±0.2 was considered.
**The threshold stays at ±0.1 for the primary analysis.** Reasons:

1. **Changing it now would be a post-hoc choice.** The threshold was set before the results. Moving it after seeing a
   BACI pH effect of about +0.11, close to the threshold, is the kind of retrospective choice the pre-specification is
   meant to prevent, whichever way it moves the conclusions.
2. **It doesn't change any pH verdict.** Both pH results (before/after and BACI, at 5 km) are *detected*, which doesn't
   depend on the threshold. Only the size comparison changes: BACI pH is about 1× the threshold at ±0.1 and about 0.5×
   at ±0.2. `sesoi_sensitivity.csv` and section 10 of `distance_model_results.ipynb` already show every estimate at
   0.5×, 0.75×, 1×, 1.5× and 2× the thresholds; 2× is ±0.2 for pH.
3. **Instrument accuracy limits single readings, not averages.** ±0.1 describes one reading. The model estimates a
   mean change over thousands of daily values from many dams, and random error averages out. **Systematic** error does
   not: calibration practice, sensors or the sampling organization changing at the time of removal could shift pH by
   about 0.1. BACI reduces this only where the same organization samples both the upstream and downstream stations.
   This is the main pH caveat.
4. **It is consistent with the other log-scale thresholds.** pH is already a log10 scale (−log10 of H⁺ activity), so
   ±0.1 pH is about a 26% change in H⁺ activity (10^0.1 = 1.26), in line with the 20–25% thresholds for the other
   log-scale measures.

**How to report pH:** say that the effect is detected, give its size in pH units, compare it with both ±0.1 and ±0.2,
and state that the threshold equals single-reading instrument accuracy and has little ecological support.

## Other design choices and caveats

- **One national value per measure.** The thresholds don't vary by region, baseline level, season or stream type,
  although the ecological importance of a change does. This is deliberate for a national screening analysis.
- **Mean shifts only.** The models estimate changes in typical (daily median) values. Thresholds on extremes,
  durations or exceedances (e.g. days below 5 mg/L DO) would need different models.
- **Relative (log-scale) thresholds** treat a 25% rise from a low baseline the same as from a high one.
- **SSC and TSS share a threshold value** but are separate measures and are never pooled (Gray et al., 2000).
- **The thresholds drive "no meaningful change" and magnitude statements**, not *detected* verdicts. Results that depend
  on the threshold are identified in section 10 of `distance_model_results.ipynb`.
- **Future refinement:** compare estimated changes with absolute criteria for each site's baseline (e.g. ecoregional
  total P, DO relative to 5 mg/L) instead of a single % change.

## References

- Abbott, K.M., Zaidel, P.A., Roy, A.H., Houle, K.M., and Nislow, K.H., 2022, Investigating impacts of small dams and
  dam removal on dissolved oxygen in streams: PLOS ONE, v. 17, no. 11, e0277647,
  https://doi.org/10.1371/journal.pone.0277647.
- Chambers, P.A., McGoldrick, D.J., Brua, R.B., Vis, C., Culp, J.M., and Benoy, G.A., 2012, Development of
  environmental thresholds for nitrogen and phosphorus in streams: Journal of Environmental Quality, v. 41, no. 1,
  p. 7–20, https://doi.org/10.2134/jeq2010.0273.
- Gray, J.R., Glysson, G.D., Turcios, L.M., and Schwarz, G.E., 2000, Comparability of suspended-sediment
  concentration and total suspended solids data: U.S. Geological Survey Water-Resources Investigations Report
  00-4191, 14 p., https://doi.org/10.3133/wri004191.
- Heck, M.P., Schultz, L.D., Hockman-Wert, D., Dinger, E.C., and Dunham, J.B., 2018, Monitoring stream
  temperatures—A guide for non-specialists: U.S. Geological Survey Techniques and Methods, book 3, chap. A25, 76 p.,
  https://doi.org/10.3133/tm3A25.
- U.S. Environmental Protection Agency, 1986, Quality criteria for water 1986: EPA 440/5-86-001 (pH 6.5–9 for
  freshwater aquatic life).
- U.S. Environmental Protection Agency, 2000–2001, Ambient water quality criteria recommendations, rivers and streams,
  by nutrient ecoregion (e.g. ecoregion II, EPA 822-B-00-015, December 2000); summary table:
  https://www.epa.gov/system/files/documents/2021-07/ecoregion-table-rivers-streams.pdf.
- U.S. Environmental Protection Agency, 2011, A field-based aquatic life benchmark for conductivity in central
  Appalachian streams: EPA/600/R-10/023F.
- U.S. Environmental Protection Agency, n.d., Indicators: Dissolved oxygen (National Aquatic Resource Surveys),
  https://www.epa.gov/national-aquatic-resource-surveys/indicators-dissolved-oxygen.
- U.S. Geological Survey, 2023, Use of multiparameter instruments for routine field measurements (ver. 1.1, June
  2025): U.S. Geological Survey Techniques and Methods, book 9, chap. A6.8, 13 p., https://doi.org/10.3133/tm9A6.8
  (accuracy: table 6.8–2; stabilization criteria: table 6.8–5).
