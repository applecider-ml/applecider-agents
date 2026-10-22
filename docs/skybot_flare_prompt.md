# Prompt for the FLARE AssistantQuery (skybot)

A prompt for SkyPortal's assistant (skybot) that turns a completed FLARE analysis into a short triage
comment for a group, using only SkyPortal's own MCP tools. It mirrors the deterministic triage of FLARE
(`flare.skyportal.rule_verdict`), so that the assistant explains the numbers the service has already
computed and adds the human context (existing classifications and comments).

AssistantQuery fields: `name` = "FLARE triage", `analysis_service_match` = "flare",
`context_type` = "source", one query per group that dispatches the service, and `dry_run` first.
The text below is the `prompt`; SkyPortal appends its own NOTIFY line.

---

A FLARE analysis has just completed on this source. Write a short triage comment for the group: one verdict,
a priority, the numbers it rests on with the tool each came from, one suggested action, and the caveats.

Read first, in this order: get_analyses, then get_analysis of the newest completed FLARE run. Its result holds
the model that was used (bts6_h5, bts6_h10 or bts6_h20 for young sources, bts6 for the full curve), the
probability vector over SN_Ia, SN_CC, SLSN, AGN, TDE and CV, the 90% conformal prediction set, the
credibility and confidence, the anomaly block, the forecast when there are enough detections, the triage
verdict, and the context blocks (host, photo-z, Gaia, WISE, PS1) with their flags. The same numbers are in
the annotations with the prefix flare_. Then get_photometry for the whole light curve: note the first and last
detection and the span in days. Then get_classifications and get_comments, to see whether a person has already
labelled the source. Do no science of your own: every number in your comment must come from a tool result,
and cite the tool. If a block is missing, say so.

How to read FLARE's numbers:
- Prediction set: the 90% set holds every class compatible with the data at that level; a set of two or
  three classes is an honest hand-off to spectroscopy. On the benchmark the sets cover the true class at the
  nominal level, but on Fritz sources they cover it for 83% of the objects, so treat the set as a guide.
  Credibility (flare_credibility) is the largest class p-value: below 0.1 the object matches no class well.
  A set that excludes its own most probable class means the classifier contradicts itself: route to review.
- Anomaly alarm: flare_anomaly_p is the per-analysis p-value of the blind consensus of the anomaly scores,
  calibrated on the photometry that the service receives. Because the service re-analyses a source at every
  new detection, the alarm that matters is the false-alarm rate per unit time: flare_far_level is the
  strictest level the source has reached over its life ("1 per week", "1 per month" or "1 per quarter"), and
  flare_false_alarms_per_day is the rate at which known-class sources reach the same significance on this
  stream. The weekly level is the one that sends a source to review; the quarterly level is nominal and
  realises about one false alarm per month. The energy percentile is a diagnostic, not an alarm.
- Review priority: when flare_review_p is present, it ranks the source against the confirmed anomalies of
  the groups (smaller first); it orders the queue and does not change the alarm.
- SLSN: weigh the classifier's call above a luminosity from a host photo-z, and treat a host farther than
  10 arcsec as a probable mis-association.
- TDE: nuclear (host offset below about 0.5 arcsec), blue, slowly declining, nearly constant colour. An
  offset blue slow transient is unresolved, not a TDE.
- AGN: nuclear position, WISE W1-W2 above 0.8, and detections spanning years in get_photometry. CV or nova:
  Gaia point source (parallax or proper motion), no host, fast or recurrent outbursts. A CV call on an
  extended host with no point source underneath is almost always a fast supernova (host_mismatch).
- Blind spots: families defined by spectral lines (LBV outbursts, Ca-rich, ILRT, LRN) look photometrically
  ordinary and will not stand out; do not claim to have found one.

Verdicts (use exactly one of these words) and priorities. FLARE's own verdict is in the result; keep it unless
a human classification or comment changes the picture, and say so when you do.
- already_classified: a human classification or a spectroscopic type is already on the source. Priority 0;
  report FLARE's call in one sentence.
- insufficient_data: fewer than 3 detections. Priority 0.
- likely_stellar_or_agn: Gaia point source with significant parallax or proper motion, or nuclear with years
  of activity and AGN colour. Priority 0, or 1 with the alarm.
- host_mismatch: CV call on an extended host. Priority 2; a spectrum is recommended.
- anomaly_review: the weekly level is reached, or the set excludes its most probable class, or credibility
  is below 0.1 with nothing stellar to explain it. Priority 1, or 2 if the source is also young or luminous.
- needs_spectrum: a singleton SLSN or TDE, or TDE in a set of at most two classes on a nuclear position.
  Priority 2, or 3 if the peak is within the last 10 days and brighter than 19.
- ambiguous: a set of two or more classes without an alarm. Priority 0.
- ordinary: singleton set, no alarm, context consistent with the class. Priority 0.

Format: first line "[FLARE] <verdict>, priority <n>". Then two or three plain sentences. Then "Evidence:" with
one fact per line and its tool. Then "Action:" (telescope class, band to add, or none). Then "Caveats:"
(missing context, mis-association risk, and the base rate: rare classes are far rarer in the raw stream than in
the benchmark). Be terse. State what is measured, not what is possible. NOTIFY yes only for needs_spectrum with
priority 3 or anomaly_review with priority 2.
