use futures_util::StreamExt;
use naori_ai::NaoriAI;
use std::io::{self, Write};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Simple Generation Example - Naori AI Library");
        println!("\nUsage:");
        println!("  generation <prompt>              - Generate with streaming");
        println!("  generation --no-stream <prompt>  - Generate without streaming");
        println!("\nExample:");
        println!("  generation \"Write a haiku about coding\"");
        println!("  generation --no-stream \"Explain Rust ownership\"");
        return Ok(());
    }

    // Parse arguments
    let (use_streaming, prompt) = if args[1] == "--no-stream" {
        if args.len() < 3 {
            return Err("Prompt required after --no-stream flag".into());
        }
        (false, args[2..].join(" "))
    } else {
        (true, args[1..].join(" "))
    };

    println!("Simple Generation Example - Naori AI Library");
    println!("Prompt: {}\n", prompt);

    // Provider selection
    let client = select_provider().await?;

    if use_streaming {
        // Streaming generation
        println!("{}: ", client.model());
        io::stdout().flush()?;

        let mut stream = client.generate_stream(&prompt).await?;
        let mut full_response = String::new();
        let mut final_usage = None;

        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| format!("Stream error: {}", e))?;
            
            if !item.content.is_empty() {
                print!("{}", item.content);
                io::stdout().flush()?;
                full_response.push_str(&item.content);
            }

            if let Some(usage) = item.usage {
                final_usage = Some(usage);
            }
            
            if item.done {
                break;
            }
        }

        println!("\n");

        // Display usage statistics if available
        if let Some(usage) = &final_usage {
            println!("Usage: {} input + {} output = {} total tokens", 
                usage.prompt_tokens.unwrap_or(0),
                usage.completion_tokens.unwrap_or(0), 
                usage.total_tokens.unwrap_or(0)
            );
        }
    } else {
        // Non-streaming generation
        println!("{}: ", client.model());
        io::stdout().flush()?;

        let response = client.generate(&prompt).await?;
        println!("{}\n", response);
    }

    Ok(())
}

fn get_api_key(env_var: &str, provider_name: &str) -> Result<String, Box<dyn std::error::Error>> {
    match std::env::var(env_var) {
        Ok(key) => {
            println!("Using {} API key from environment variable", provider_name);
            Ok(key)
        }
        Err(_) => {
            print!("Enter {} API key: ", provider_name);
            io::stdout().flush()?;
            
            let mut input_key = String::new();
            io::stdin().read_line(&mut input_key)?;
            let input_key = input_key.trim().to_string();

            if input_key.is_empty() {
                return Err("API key cannot be empty".into());
            }
            Ok(input_key)
        }
    }
}

fn get_user_choice(prompt: &str) -> Result<usize, Box<dyn std::error::Error>> {
    print!("{}", prompt);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    input.trim().parse().map_err(|_| "Invalid number".into())
}

async fn select_ollama_model() -> Result<NaoriAI, Box<dyn std::error::Error>> {
    println!("\nConnecting to Ollama...");
    let temp_client = NaoriAI::ollama("http://localhost:11434".to_string(), "temp".to_string());
    
    let models = temp_client.list_local_models().await.map_err(|e| {
        println!("Failed to connect to Ollama: {}", e);
        println!("Make sure Ollama is running on http://localhost:11434");
        e
    })?;

    if models.is_empty() {
        println!("No models available. Please pull a model first using 'ollama pull <model_name>'");
        return Err("No models available".into());
    }

    println!("\nAvailable models:");
    for (i, model) in models.iter().enumerate() {
        println!("{}. {} ({:.1} GB)", i + 1, model.name, model.size as f64 / 1_073_741_824.0);
    }

    let choice = get_user_choice(&format!("Select model (1-{}): ", models.len()))?;
    if choice == 0 || choice > models.len() {
        return Err("Invalid model selection".into());
    }

    let selected_model = &models[choice - 1];
    println!("\nSelected: {}", selected_model.name);

    Ok(NaoriAI::ollama("http://localhost:11434".to_string(), selected_model.name.clone()))
}

async fn select_cloud_model<F>(
    provider_name: &str,
    env_var: &str,
    constructor: F,
    model_filter: Option<fn(&naori_ai::core::MonoModel) -> bool>,
) -> Result<NaoriAI, Box<dyn std::error::Error>>
where
    F: Fn(String, String) -> NaoriAI,
{
    let api_key = get_api_key(env_var, provider_name)?;
    
    println!("\nFetching available models...");
    let temp_client = constructor(api_key.clone(), "temp".to_string());
    
    let models = temp_client.get_available_models().await.map_err(|e| {
        println!("Failed to fetch {} models: {}", provider_name, e);
        println!("Please check your API key and internet connection");
        e
    })?;

    if models.is_empty() {
        return Err("No models available".into());
    }

    let filtered_models: Vec<_> = if let Some(filter) = model_filter {
        models.into_iter().filter(filter).collect()
    } else {
        models
    };

    if filtered_models.is_empty() {
        return Err("No suitable models available".into());
    }

    println!("\nAvailable {} models:", provider_name);
    for (i, model) in filtered_models.iter().enumerate() {
        println!("{}. {} ({})", i + 1, model.name, model.id);
    }

    if provider_name == "OpenRouter" && filtered_models.iter().any(|m| m.id == "custom") {
        println!("\nNote: Select Custom Model to manually enter any OpenRouter model ID");
    }

    let choice = get_user_choice(&format!("Select model (1-{}): ", filtered_models.len()))?;
    if choice == 0 || choice > filtered_models.len() {
        return Err("Invalid model selection".into());
    }

    let selected_model = &filtered_models[choice - 1];
    
    let final_model_id = if selected_model.id == "custom" {
        print!("Enter OpenRouter model ID (e.g., anthropic/claude-sonnet-4): ");
        io::stdout().flush()?;
        let mut custom_model = String::new();
        io::stdin().read_line(&mut custom_model)?;
        let custom_model = custom_model.trim().to_string();
        if custom_model.is_empty() {
            return Err("Model ID cannot be empty".into());
        }
        println!("Selected custom model: {}", custom_model);
        custom_model
    } else {
        println!("Selected: {}", selected_model.name);
        selected_model.id.clone()
    };

    Ok(constructor(api_key, final_model_id))
}

async fn select_provider() -> Result<NaoriAI, Box<dyn std::error::Error>> {
    println!("Select AI Provider:");
    println!("1. Ollama (local & cloud)");
    println!("2. Anthropic (cloud)");
    println!("3. OpenAI (cloud)");
    println!("4. OpenRouter (cloud)");
    
    let choice = get_user_choice("Enter choice (1-4): ")?;

    match choice {
        1 => select_ollama_model().await,
        2 => select_cloud_model("Anthropic", "ANTHROPIC_API_KEY", NaoriAI::anthropic, None).await,
        3 => {
            let openai_filter = |m: &naori_ai::core::MonoModel| m.id.contains("gpt") || m.id.contains("o1");
            select_cloud_model("OpenAI", "OPENAI_API_KEY", NaoriAI::openai, Some(openai_filter)).await
        }
        4 => select_cloud_model("OpenRouter", "OPENROUTER_API_KEY", NaoriAI::openrouter, None).await,
        _ => {
            println!("Invalid choice. Exiting.");
            Err("Invalid provider selection".into())
        }
    }
}
