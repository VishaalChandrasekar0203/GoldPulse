SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS series_catalog (
    series_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    description VARCHAR,
    unit VARCHAR,
    frequency VARCHAR,
    source VARCHAR,
    source_url VARCHAR,
    license VARCHAR,
    publication_lag_days INTEGER,
    category VARCHAR,
    status VARCHAR,
    notes VARCHAR,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS observations (
    series_id VARCHAR,
    obs_date DATE,
    value DOUBLE,
    available_at DATE,
    frequency VARCHAR,
    source VARCHAR,
    ingested_at TIMESTAMP,
    run_id VARCHAR,
    is_derived BOOLEAN,
    quality_flag VARCHAR,
    provenance JSON,
    PRIMARY KEY (series_id, obs_date)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    status VARCHAR,
    records_fetched INTEGER,
    records_inserted INTEGER,
    records_updated INTEGER,
    skipped_sources JSON,
    errors JSON,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS anomalies (
    series_id VARCHAR,
    obs_date DATE,
    value DOUBLE,
    zscore DOUBLE,
    reason VARCHAR,
    detected_at TIMESTAMP,
    PRIMARY KEY (series_id, obs_date, reason)
);

CREATE TABLE IF NOT EXISTS feature_runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    n_rows INTEGER,
    n_features INTEGER,
    cutoff_date DATE,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id VARCHAR PRIMARY KEY,
    model_name VARCHAR,
    horizon INTEGER,
    trained_at TIMESTAMP,
    cutoff_date DATE,
    metrics JSON,
    params JSON,
    artifact_path VARCHAR,
    is_champion BOOLEAN,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id VARCHAR PRIMARY KEY,
    pipeline_version VARCHAR,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    n_experiments INTEGER,
    comparison_json JSON,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id VARCHAR,
    generated_at TIMESTAMP,
    model_name VARCHAR,
    model_version VARCHAR,
    target_series VARCHAR,
    cutoff_date DATE,
    horizon_calendar_days INTEGER,
    horizon_trading_days INTEGER,
    target_date DATE,
    target_is_trading_day BOOLEAN,
    point DOUBLE,
    lower DOUBLE,
    upper DOUBLE,
    interval_low DOUBLE,
    interval_high DOUBLE,
    data_cutoff TIMESTAMP,
    notes VARCHAR,
    PRIMARY KEY (forecast_id, target_date, horizon_calendar_days)
);

CREATE TABLE IF NOT EXISTS system_log (
    ts TIMESTAMP,
    level VARCHAR,
    component VARCHAR,
    message VARCHAR
);

CREATE TABLE IF NOT EXISTS forecast_outcomes (
    outcome_id VARCHAR PRIMARY KEY,
    forecast_id VARCHAR,
    source VARCHAR,
    model_name VARCHAR,
    model_version VARCHAR,
    target_series VARCHAR,
    cutoff_date DATE,
    target_date DATE,
    horizon_calendar_days INTEGER,
    interval_method VARCHAR,
    origin_close DOUBLE,
    point DOUBLE,
    lower DOUBLE,
    upper DOUBLE,
    realized DOUBLE,
    abs_error DOUBLE,
    signed_error DOUBLE,
    covered BOOLEAN,
    winkler DOUBLE,
    direction_correct BOOLEAN,
    scored_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    severity VARCHAR,
    kind VARCHAR,
    horizon INTEGER,
    series_id VARCHAR,
    message VARCHAR,
    payload JSON
);

CREATE TABLE IF NOT EXISTS quality_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    captured_at TIMESTAMP,
    series_id VARCHAR,
    n_obs INTEGER,
    start_date DATE,
    end_date DATE,
    last_available_at DATE,
    n_missing_provenance INTEGER,
    n_timestamp_issues INTEGER,
    n_flagged INTEGER,
    n_gaps INTEGER,
    stale BOOLEAN,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id VARCHAR PRIMARY KEY,
    ts TIMESTAMP,
    actor VARCHAR,
    kind VARCHAR,
    ref_id VARCHAR,
    run_id VARCHAR,
    payload JSON
);

CREATE TABLE IF NOT EXISTS source_health (
    check_id VARCHAR PRIMARY KEY,
    checked_at TIMESTAMP,
    series_id VARCHAR,
    status VARCHAR,
    last_obs DATE,
    lag_days INTEGER,
    critical BOOLEAN,
    message VARCHAR
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    status VARCHAR,
    ingest_status VARCHAR,
    forecasted BOOLEAN,
    n_forecasts INTEGER,
    n_outcomes_scored INTEGER,
    backup_path VARCHAR,
    notes VARCHAR,
    payload JSON
);
"""
