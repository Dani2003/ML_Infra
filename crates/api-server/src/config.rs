use std::{env, net::SocketAddr, time::Duration};

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub bind: SocketAddr,
    pub queue_capacity: usize,
    pub max_active_requests: usize,
    pub max_prompt_chars: usize,
    pub max_new_tokens: usize,
    pub request_timeout: Duration,
    pub decode_step_interval: Duration,
}

impl ServerConfig {
    pub fn from_env() -> Result<Self, String> {
        let bind = value("INFER_RS_BIND", "127.0.0.1:3000")?.parse()
            .map_err(|e| format!("INFER_RS_BIND must be a socket address: {e}"))?;
        let config = Self {
            bind,
            queue_capacity: number("INFER_RS_QUEUE_CAPACITY", 128)?,
            max_active_requests: number("INFER_RS_MAX_ACTIVE_REQUESTS", 32)?,
            max_prompt_chars: number("INFER_RS_MAX_PROMPT_CHARS", 8192)?,
            max_new_tokens: number("INFER_RS_MAX_NEW_TOKENS", 256)?,
            request_timeout: Duration::from_secs(number("INFER_RS_REQUEST_TIMEOUT_SECS", 120)?),
            decode_step_interval: Duration::from_millis(number("INFER_RS_STEP_INTERVAL_MS", 10)?),
        };
        if config.queue_capacity == 0 { return Err("INFER_RS_QUEUE_CAPACITY must be greater than zero".into()); }
        if config.max_active_requests == 0 { return Err("INFER_RS_MAX_ACTIVE_REQUESTS must be greater than zero".into()); }
        if config.max_prompt_chars == 0 { return Err("INFER_RS_MAX_PROMPT_CHARS must be greater than zero".into()); }
        if config.max_new_tokens == 0 { return Err("INFER_RS_MAX_NEW_TOKENS must be greater than zero".into()); }
        if config.request_timeout.is_zero() { return Err("INFER_RS_REQUEST_TIMEOUT_SECS must be greater than zero".into()); }
        if config.request_timeout > Duration::from_secs(86_400) {
            return Err("INFER_RS_REQUEST_TIMEOUT_SECS must not exceed 86400".into());
        }
        Ok(config)
    }
}

fn value(name: &str, default: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) => Ok(value),
        Err(env::VarError::NotPresent) => Ok(default.to_owned()),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{name} is not valid Unicode")),
    }
}

fn number<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String>
where T::Err: std::fmt::Display {
    match env::var(name) {
        Ok(raw) => raw.parse().map_err(|e| format!("invalid {name}: {e}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{name} is not valid Unicode")),
    }
}
