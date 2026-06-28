//! Registry of configured providers.

use super::Provider;

/// Holds every configured provider and remembers which is the default.
pub struct ProviderRegistry {
    providers: Vec<Box<dyn Provider>>,
    default: String,
}

impl ProviderRegistry {
    /// Build a registry. `default` must name one of `providers`.
    pub fn new(providers: Vec<Box<dyn Provider>>, default: String) -> Self {
        Self { providers, default }
    }

    /// Look up a provider by name.
    pub fn get(&self, name: &str) -> Option<&dyn Provider> {
        self.providers
            .iter()
            .find(|p| p.name() == name)
            .map(|p| p.as_ref())
    }

    /// The configured default provider.
    pub fn default(&self) -> &dyn Provider {
        self.get(&self.default)
            .or_else(|| self.providers.first().map(|p| p.as_ref()))
            .expect("registry has at least one provider")
    }

    /// Name of the default provider.
    pub fn default_name(&self) -> &str {
        &self.default
    }

    /// First provider that declares support for `model`.
    pub fn supporting(&self, model: &str) -> Option<&dyn Provider> {
        self.providers
            .iter()
            .find(|p| p.supports_model(model))
            .map(|p| p.as_ref())
    }

    /// Iterate over all providers.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Provider> {
        self.providers.iter().map(|p| p.as_ref())
    }
}
