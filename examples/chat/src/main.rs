use futures_util::StreamExt;
use naori_ai::{Message, NaoriAI};
use naori_ai_macros::tool;
use std::io::{self, Write};
use colored::*;

#[tool]
/// Get the current weather for a given location
fn get_weather(location: String) -> String {
    format!("Weather in {}: 72°F and sunny", location)
}

#[tool]
/// Generate a secure password with specified length
fn generate_password(length: usize) -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                            abcdefghijklmnopqrstuvwxyz\
                            0123456789)(*&^%$#@!~";
    
    let mut rng = rand::rng();
    (0..length)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Naori AI Rust Library");
    println!("This demonstrates streaming chat with optional tool calling");

    // Provider selection
    let mut client = select_provider().await?;

    // the rest of the code below works the same regardless of provider
    
    // Add tools (optional)
    client.add_tool(get_weather_tool()).await?;
    client.add_tool(generate_password_tool()).await?;

    // Show fallback mode status
    if client.is_fallback_mode().await {
        println!("Using fallback mode for tool calling (model doesn't support native tools)");
    } else {
        println!("Using native tool calling support");
    }

    let mut messages = Vec::new();

    println!("Type your messages (or quit to exit):");

    loop {
        print!("\nYou: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "quit" || input == "exit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        messages.push(Message {
            role: "user".to_string(),
            content: input.to_string(),
            images: None,
            tool_calls: None,
        });

        print!("{}: ", client.model());
        io::stdout().flush()?;

        let mut stream = client.send_chat_request(&messages).await?;
        let mut full_response = String::new();
        let mut tool_calls = None;
        let mut final_usage = None;

        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| format!("Stream error: {}", e))?;
            
            if !item.content.is_empty() {
                print!("{}", item.content);
                io::stdout().flush()?;
                full_response.push_str(&item.content);
            }
            
            if let Some(tc) = item.tool_calls {
                tool_calls = Some(tc);
            }

            if let Some(usage) = item.usage {
                final_usage = Some(usage);
            }
            
            if item.done {
                break;
            }
        }


        // Display usage statistics if available
        if let Some(usage) = &final_usage {
            let cost_display = if let Some(cost) = usage.cost_usd {
                if cost > 0.0 {
                    format!(" (${:.6})", cost)
                } else {
                    " (free)".to_string()
                }
            } else {
                "".to_string()
            };
            println!("\n{}", format!("Usage: {} input + {} output = {} total tokens{}", 
                usage.prompt_tokens.unwrap_or(0),
                usage.completion_tokens.unwrap_or(0), 
                usage.total_tokens.unwrap_or(0),
                cost_display
            ).truecolor(128, 128, 128));
        }

        // Add assistant response with tool calls to conversation
        messages.push(Message {
            role: "assistant".to_string(),
            content: full_response,
            images: None,
            tool_calls: tool_calls.clone(), // Include tool calls in the conversation history
        });

        // Handle tool calls
        if let Some(ref tc) = tool_calls {
            // Tool execution status (remove these prints for silent operation)
            for tool_call in tc {
                println!("\n{}", format!("Using {} tool...", tool_call.function.name).truecolor(169, 169, 169));
            }
            
            let tool_responses = client.handle_tool_calls(tc.clone()).await;
            
            // Show tool results
            for (tool_call, response) in tc.iter().zip(tool_responses.iter()) {
                // Extract clean result from encoded format for display
                let clean_result = if response.content.starts_with("TOOL_RESULT:") {
                    // Parse "TOOL_RESULT:tool_id:actual_result" and extract actual_result
                    let parts: Vec<&str> = response.content.splitn(3, ':').collect();
                    if parts.len() == 3 {
                        parts[2]
                    } else {
                        &response.content
                    }
                } else {
                    &response.content
                };
                println!("{}", format!("{} called, result: {}", tool_call.function.name, clean_result).green());
            }
            
            messages.extend(tool_responses);

            // Continue conversation after tool execution  
            print!("{}: ", client.model());
            io::stdout().flush()?;
            let mut tool_stream = client.send_chat_request(&messages).await?;
            let mut final_response = String::new();
            let mut tool_usage = None;
            while let Some(item) = tool_stream.next().await {
                let item = item.map_err(|e| format!("Stream error: {}", e))?;
                if !item.content.is_empty() {
                    print!("{}", item.content);
                    io::stdout().flush()?;
                    final_response.push_str(&item.content);
                }
                if let Some(usage) = item.usage {
                    tool_usage = Some(usage);
                }
                if item.done {
                    break;
                }
            }


            // Display tool follow-up usage
            if let Some(usage) = &tool_usage {
                let cost_display = if let Some(cost) = usage.cost_usd {
                    if cost > 0.0 {
                        format!(" (${:.6})", cost)
                    } else {
                        " (free)".to_string()
                    }
                } else {
                    "".to_string()
                };
                println!("\n{}", format!("Tool follow-up usage: {} input + {} output = {} total tokens{}", 
                    usage.prompt_tokens.unwrap_or(0),
                    usage.completion_tokens.unwrap_or(0), 
                    usage.total_tokens.unwrap_or(0),
                    cost_display
                ).truecolor(128, 128, 128));
            }
            
            // Add the final assistant response to conversation
            messages.push(Message {
                role: "assistant".to_string(),
                content: final_response,
                images: None,
                tool_calls: None,
            });
        }

        println!();
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

    println!("\nAvailable local models:");
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
    println!("1. Ollama (local)");
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