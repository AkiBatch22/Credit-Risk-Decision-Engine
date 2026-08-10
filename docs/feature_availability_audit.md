# Feature Availability Audit

This audit answers the production question: **where does each value come from when a completely new applicant submits an application?** The persisted training metadata also records a classification and inference-source answer for every column found in the training CSV.

## Deployable applicant-provided fields

| Internal field | New-application source | Treatment |
|---|---|---|
| `DAYS_BIRTH` | Applicant enters age | Converted from completed years; negative days are internal only |
| `DAYS_EMPLOYED` | Applicant enters employment duration | Converted from years; negative days are internal only |
| `CNT_FAM_MEMBERS` | Applicant enters household size | Validated numeric input |
| `CNT_CHILDREN` | Applicant enters number of children | Validated numeric input |
| `AMT_INCOME_TOTAL` | Applicant enters annual income | Validated numeric input |
| `AMT_CREDIT` | Applicant enters requested credit | Validated numeric input and illustrative EAD |
| `AMT_ANNUITY` | Applicant enters recorded repayment obligation | Validated numeric input; no unsupported frequency claim |
| `AMT_GOODS_PRICE` | Applicant enters associated goods/asset price | Validated numeric input |
| `NAME_CONTRACT_TYPE` | Selected credit product | Contract-controlled category |
| `NAME_INCOME_TYPE` | Applicant declares income category | Optional contract-controlled category |
| `NAME_HOUSING_TYPE` | Applicant declares housing situation | Optional contract-controlled category |
| `FLAG_OWN_CAR` | Applicant declares car ownership | Yes/No mapped to historical `Y`/`N` values |
| `FLAG_OWN_REALTY` | Applicant declares property ownership | Yes/No mapped to historical `Y`/`N` values |

These fields are centralized in `src/components/feature_contract.py`. Training selects them explicitly before feature engineering, so unused CSV columns cannot enter the deployable model.

## Deployable system-derived fields

All production-derived values are deterministic functions of the applicant-provided fields: `DAYS_EMPLOYED_ANOMALY`, `DAYS_EMPLOYED_CLEAN`, `AGE_YEARS`, `EMPLOYMENT_YEARS`, `EMPLOYMENT_AGE_RATIO`, `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_ANNUITY_RATIO`, `CREDIT_GOODS_RATIO`, `INCOME_PER_PERSON`, `CREDIT_PER_PERSON`, `ANNUITY_PER_PERSON`, and `CHILDREN_FAMILY_RATIO`.

Ratios use zero-safe division, the `365243` employment sentinel is converted to missing with an explicit flag, and infinities become missing. No feature uses `TARGET`.

## External or unavailable

`EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` are strong historical predictors, but the dataset anonymizes their meaning and this repository cannot reproduce their generation. They and `EXT_SOURCE_MEAN`, `EXT_SOURCE_MIN`, `EXT_SOURCE_MAX`, `EXT_SOURCE_STD`, and `EXT_SOURCE_COUNT` are benchmark-only and excluded from deployable preprocessing.

Credit-bureau request counts (`AMT_REQ_CREDIT_BUREAU_HOUR`, `DAY`, `WEEK`, `MON`, `QRT`, and `YEAR`) are also excluded because no bureau integration or reproducible provider exists in this project.

## Historical identifiers

`SK_ID_CURR` identifies a historical Home Credit row. It remains useful in research and is preserved as `source_record_id` during historical batch scoring. It is never selected by the feature contract. New requests receive a server-generated `application_id` matching `APP-[0-9A-F]{10}`; that identifier is attached only after the model input has been constructed.

## Sensitive or questionable proxies

The recommended deployment contract excludes `CODE_GENDER`, `NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE`, and `OCCUPATION_TYPE`. They can encode protected characteristics or strong socioeconomic proxies and are not justified here by a documented lending policy, jurisdictional legal review, or fairness analysis.

Age, household composition, income type, housing, and asset ownership can also be sensitive or proxy-bearing depending on jurisdiction. Their inclusion here demonstrates an inference-available contract, not legal suitability. A real deployment would require necessity analysis, subgroup calibration and error testing, adverse-impact review, accessible alternatives for missing information, and an appeal process.

## Questionable or non-deployable historical fields

The following historical families are excluded because this project has no implemented, validated source for a completely new application:

- Application context or workflow metadata: `NAME_TYPE_SUITE`, `WEEKDAY_APPR_PROCESS_START`, `HOUR_APPR_PROCESS_START`, `ORGANIZATION_TYPE`.
- Regional/location and mobility proxies: `REGION_POPULATION_RELATIVE`, `REGION_RATING_CLIENT`, `REGION_RATING_CLIENT_W_CITY`, `REG_REGION_NOT_LIVE_REGION`, `REG_REGION_NOT_WORK_REGION`, `LIVE_REGION_NOT_WORK_REGION`, `REG_CITY_NOT_LIVE_CITY`, `REG_CITY_NOT_WORK_CITY`, `LIVE_CITY_NOT_WORK_CITY`.
- Administrative/account-history day offsets: `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`.
- Contact-channel telemetry: `FLAG_MOBIL`, `FLAG_EMP_PHONE`, `FLAG_WORK_PHONE`, `FLAG_CONT_MOBILE`, `FLAG_PHONE`, `FLAG_EMAIL`.
- Social-circle history: `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE`, `OBS_60_CNT_SOCIAL_CIRCLE`, `DEF_60_CNT_SOCIAL_CIRCLE`.
- Document workflow flags: `FLAG_DOCUMENT_2` through `FLAG_DOCUMENT_21`.
- Vehicle detail: `OWN_CAR_AGE` (ownership itself remains a declared contract field).
- Property/building records: `APARTMENTS_*`, `BASEMENTAREA_*`, `YEARS_BEGINEXPLUATATION_*`, `YEARS_BUILD_*`, `COMMONAREA_*`, `ELEVATORS_*`, `ENTRANCES_*`, `FLOORSMAX_*`, `FLOORSMIN_*`, `LANDAREA_*`, `LIVINGAPARTMENTS_*`, `LIVINGAREA_*`, `NONLIVINGAPARTMENTS_*`, `NONLIVINGAREA_*`, `FONDKAPREMONT_MODE`, `HOUSETYPE_MODE`, `TOTALAREA_MODE`, `WALLSMATERIAL_MODE`, and `EMERGENCYSTATE_MODE`.

These fields may be legitimate in a differently specified system with documented upstream services. In this repository, median/mode imputation would only hide a permanent training-serving mismatch, so they are excluded instead.

## Research versus deployment

The full-feature benchmark remains useful for understanding the dataset and estimating the performance ceiling under historical feature availability. The deployable model intentionally sacrifices any performance associated with unavailable features so that every production feature can be reproduced for a new applicant. Neither profile establishes suitability for real-world lending.
