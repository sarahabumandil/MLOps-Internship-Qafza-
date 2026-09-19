# Great Expectations

This project defines its expectation suite (`order_intake_suite`) as code
in `src/data/validation.py`, using an **ephemeral** (in-memory) Great
Expectations context rather than a filesystem-backed GE project. That
means:

- The suite is versioned in git like any other source file — the
  reviewable, diffable definition of "what a valid order looks like" is
  `NUMERIC_BOUNDS`, `VALID_BR_STATES`, and `REQUIRED_NOT_NULL` at the top
  of `src/data/validation.py`.
- No `great_expectations.yml` / `uncommitted/` project scaffolding is
  needed at runtime — `_build_validation_definition()` builds the context,
  data source, and suite fresh (and caches it) the first time the service
  validates an order.
- Every prediction request runs through `enforce_validation_policy()`
  before it reaches the model.

If you'd rather have a full filesystem GE project (e.g. to use GE's Data
Docs HTML reports), run `great_expectations init` here and port the
expectations from `validate_orders()` into a persisted suite — the
`ge_root` path in `config/config.yaml` already points at this folder for
that purpose.
