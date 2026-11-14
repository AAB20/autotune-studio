# ADR-0006: Malformed-record side channel on connectors

Status: accepted (2026-11-14)

`Connector.iter_batches` gains an optional `on_malformed` callback. Connectors report records they cannot
parse (bad JSON lines, non-object items) with their location (byte offset / item index); the validator counts
them in `StreamStats.malformed_rows`. This keeps the "never drop silently" rule without forcing malformed data
into Arrow batches. `connectors.base.iter_with_sink` adapts connectors that do not accept the argument.
