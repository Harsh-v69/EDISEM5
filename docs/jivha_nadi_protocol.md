# Jivha (tongue/nail) and Nadi (pulse) module: protocol draft

**Status: not started. No data has been collected.** This module needs a clinical partner (for example a BAMS college or Ayurveda clinic) and ethics-committee approval before any image or signal from a person is collected. This document is a *proposal for the partner to review and correct*: every clinical detail below is a suggestion, not established practice. The partner-independent code that exists today is `ayurveda_kg/vision_nadi/agreement.py` (Cohen's kappa, tested).

## 1. What the module would do (and not do)
- Input: a standardised tongue photo, nail photo, and/or a pulse signal. Output: a *decision-support* estimate of a dosha profile with an uncertainty, which becomes a Patient node used to personalise the project's graph queries.
- It does **not** diagnose disease and must never be described as a validated medical device. The only claim the project would make is agreement with independent practitioners (kappa), with the practitioners' own agreement reported as context.

## 2. Gate before any collection (all must be "yes")
- [ ] A named clinical partner and principal investigator.
- [ ] Ethics committee approval for image/signal collection, including consent text in the participants' language.
- [ ] A data-protection plan agreed with the partner's institution and checked against the applicable Indian data-protection law (to be confirmed by the institution's legal or ethics office): what is stored, where, who can access it, retention, and how withdrawal is honoured.
- [ ] No faces or identifying features in stored images; participant identifiers kept separate from images.
- [ ] A pre-agreed analysis plan (below) written before looking at any results.

## 3. Capture protocol (proposed; the partner should adjust)
- **Tongue (Jivha):** same room, diffuse stable lighting, a colour reference card in frame, fixed distance and angle (a printed on-screen guide), phone model and settings recorded, tongue extended relaxed for a few seconds. Record pre-conditions the partner considers relevant (for example time since eating, drinking, tongue cleaning or smoking).
- **Nails (Nakha):** hands at rest on a neutral background, same lighting and colour card, all ten nails in one frame if practical.
- **Pulse (Nadi):** signal acquisition needs a sensor (for example a photoplethysmography or piezo pulse sensor) with documented sampling rate and placement; the manual pulse reading by the practitioner is recorded separately and blind to the sensor output.
- Pilot first: 50 to 100 captures to check the protocol yields consistent, usable data before scaling up.

## 4. Annotation guide (proposed)
- At least **two practitioners label each item independently**, blind to each other and to the model. The partner defines the label sets; suggested starting categories: coating (none/thin/thick), colour category, moisture, and a dosha-imbalance indicator.
- A label "uncertain / image unusable" is always allowed and is reported, not hidden.
- Labels are practitioner impressions, never confirmed disease.
- Check `validate_annotations` (duplicates, unknown labels, items with a single rater) before computing anything.

## 5. Validation plan (to be fixed in advance)
- Pairwise Cohen's kappa between practitioners (`pairwise_kappa`): this is the ceiling on how well any model can agree and is reported first.
- Model versus each practitioner: kappa, accuracy and F1 per category. Patient-level (not image-level) splits so one person never appears in train and test.
- Heavy augmentation and honest reporting of the small dataset (a student project will likely have a few hundred images at most).
- Public proxy data (for example the TCM-Tongue dataset, arXiv 2507.18288) may be used only to test that the pipeline runs; **no Ayurvedic accuracy claim is made from proxy data**.

## 6. Claims policy
Allowed: "agreement with practitioners, measured by kappa, on N participants at site S". Not allowed: diagnosis, screening or treatment claims; any wording implying clinical validation.
