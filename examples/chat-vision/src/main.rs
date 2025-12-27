use futures_util::StreamExt;
use naori_ai::{Message, NaoriAI};
use std::io::{self, Write};
use std::env;

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

    println!("\nAvailable models (all models support vision):");
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

async fn select_cloud_vision_model<F>(
    provider_name: &str,
    env_var: &str,
    constructor: F,
    vision_filter: fn(&naori_ai::core::MonoModel) -> bool,
    fallback_filter: Option<fn(&naori_ai::core::MonoModel) -> bool>,
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

    // First try vision-specific filter
    let vision_models: Vec<_> = models.iter().filter(|m| vision_filter(m)).cloned().collect();

    let (filtered_models, model_type) = if vision_models.is_empty() {
        if let Some(fallback) = fallback_filter {
            let fallback_models: Vec<_> = models.into_iter().filter(fallback).collect();
            if fallback_models.is_empty() {
                return Err(format!("No suitable {} models available", provider_name).into());
            }
            println!("No vision-specific models found, showing all suitable models:");
            (fallback_models, "suitable")
        } else {
            return Err(format!("No vision-capable {} models available", provider_name).into());
        }
    } else {
        (vision_models, "vision")
    };

    println!("\nAvailable {} {} models:", provider_name, model_type);
    for (i, model) in filtered_models.iter().enumerate() {
        println!("{}. {} ({})", i + 1, model.name, model.id);
    }

    if provider_name == "OpenRouter" && filtered_models.iter().any(|m| m.id == "custom") {
        println!("\nNote: Select 'Custom Model' to manually enter any OpenRouter vision model ID");
    }

    let choice = get_user_choice(&format!("Select model (1-{}): ", filtered_models.len()))?;
    if choice == 0 || choice > filtered_models.len() {
        return Err("Invalid model selection".into());
    }

    let selected_model = &filtered_models[choice - 1];
    
    let final_model_id = if selected_model.id == "custom" {
        print!("Enter OpenRouter vision model ID (e.g., anthropic/claude-sonnet-4): ");
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
        2 => {
            // All Anthropic models support vision
            let anthropic_filter = |_: &naori_ai::core::MonoModel| true;
            select_cloud_vision_model("Anthropic", "ANTHROPIC_API_KEY", NaoriAI::anthropic, anthropic_filter, None).await
        }
        3 => {
            // OpenAI vision models: GPT-4 with vision or GPT-4o
            let vision_filter = |m: &naori_ai::core::MonoModel| {
                m.id.contains("gpt-4") && (m.id.contains("vision") || m.id.contains("gpt-4o"))
            };
            let fallback_filter = |m: &naori_ai::core::MonoModel| {
                m.id.contains("gpt-4") || m.id.contains("o1")
            };
            select_cloud_vision_model("OpenAI", "OPENAI_API_KEY", NaoriAI::openai, vision_filter, Some(fallback_filter)).await
        }
        4 => {
            // OpenRouter vision models: GPT-4 vision, Claude, Gemini
            let vision_filter = |m: &naori_ai::core::MonoModel| {
                let id_lower = m.id.to_lowercase();
                (id_lower.contains("gpt-4") && (id_lower.contains("vision") || id_lower.contains("gpt-4o"))) ||
                id_lower.contains("claude") ||
                id_lower.contains("gemini") ||
                m.id == "custom"
            };
            select_cloud_vision_model("OpenRouter", "OPENROUTER_API_KEY", NaoriAI::openrouter, vision_filter, None).await
        }
        _ => {
            println!("Invalid choice. Exiting.");
            Err("Invalid provider selection".into())
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Chat Vision Example - Naori AI Library");
        println!("\nUsage:");
        println!("  chat-vision <image_path>   - Analyze image and start chat");
        return Ok(());
    }

    let image_path = &args[1];

    println!("Chat Vision Example - Naori AI Library");
    println!("Analyzing image: {}\n", image_path);

    // Provider selection
    let client = select_provider().await?;

    // Encode image for conversation history
    let encoded_image = client.encode_image_file(image_path).await?;
    
    let mut messages = vec![
        Message {
            role: "user".to_string(),
            content: "What do you see in this image?".to_string(),
            images: Some(vec![encoded_image]),
            tool_calls: None,
        }
    ];

    // Send initial image analysis request
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
        println!("\n{}", format!("Usage: {} input + {} output = {} total tokens", 
            usage.prompt_tokens.unwrap_or(0),
            usage.completion_tokens.unwrap_or(0), 
            usage.total_tokens.unwrap_or(0)
        ));
    }

    // Add assistant response to conversation
    messages.push(Message {
        role: "assistant".to_string(),
        content: full_response,
        images: None,
        tool_calls: tool_calls.clone(),
    });

    // Handle tool calls if any
    if let Some(ref tc) = tool_calls {
        let tool_responses = client.handle_tool_calls(tc.clone()).await;
        messages.extend(tool_responses);
        
        // Continue conversation after tool execution  
        print!("{}: ", client.model());
        io::stdout().flush()?;
        let mut tool_stream = client.send_chat_request(&messages).await?;
        let mut final_response = String::new();
        while let Some(item) = tool_stream.next().await {
            let item = item.map_err(|e| format!("Stream error: {}", e))?;
            if !item.content.is_empty() {
                print!("{}", item.content);
                io::stdout().flush()?;
                final_response.push_str(&item.content);
            }
            if item.done {
                break;
            }
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
            
            if item.done {
                break;
            }
        }

        // Add assistant response to conversation
        messages.push(Message {
            role: "assistant".to_string(),
            content: full_response,
            images: None,
            tool_calls: tool_calls.clone(),
        });

        // Handle tool calls if any
        if let Some(ref tc) = tool_calls {
            let tool_responses = client.handle_tool_calls(tc.clone()).await;
            messages.extend(tool_responses);
            
            // Continue conversation after tool execution  
            print!("{}: ", client.model());
            io::stdout().flush()?;
            let mut tool_stream = client.send_chat_request(&messages).await?;
            let mut final_response = String::new(); 
            while let Some(item) = tool_stream.next().await {
                let item = item.map_err(|e| format!("Stream error: {}", e))?;
                if !item.content.is_empty() {
                    print!("{}", item.content);
                    io::stdout().flush()?;
                    final_response.push_str(&item.content);
                }
                if item.done {
                    break;
                }
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