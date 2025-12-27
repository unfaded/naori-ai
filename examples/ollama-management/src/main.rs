use futures_util::StreamExt;
use naori_ai::NaoriAI;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Ollama Management Example - Naori AI Library");
        println!("This demonstrates Ollama-specific model management operations");
        println!("\nUsage:");
        println!("  cargo run list                    - List local & cloud models");
        println!("  cargo run pull <model>            - Download model with progress");
        println!("  cargo run info <model>            - Show detailed model information");
        println!("  cargo run generate <model> <text> - Simple text generation test");
        println!("\nExample:");
        println!("  cargo run list");
        println!("  cargo run pull llama3:8b");
        println!("  cargo run info qwen3-coder:30b");
        println!("  cargo run generate llama3:8b \"Write a haiku\"");
        return Ok(());
    }

    let command = &args[1];

    match command.as_str() {
        "list" => {
            println!("Listing Ollama models...\n");
            
            let client = NaoriAI::ollama("http://localhost:11434".to_string(), "".to_string());
            let models = client.list_local_models().await?;
            
            if models.is_empty() {
                println!("No models found. Use 'cargo run pull <model>' to download one.");
            } else {
                println!("Models:");
                for model in models {
                    let size_gb = model.size as f64 / 1_073_741_824.0;
                    println!("- {} ({:.1} GB)", model.name, size_gb);
                }
            }
        }

        "pull" => {
            if args.len() < 3 {
                println!("Usage: cargo run pull <model_name>");
                println!("Example: cargo run pull llama3:8b");
                return Ok(());
            }

            let model_name = &args[2];
            println!("Downloading model: {}\n", model_name);

            let client = NaoriAI::ollama("http://localhost:11434".to_string(), "".to_string());
            let mut stream = client.pull_model_stream(model_name).await?;

            while let Some(progress) = stream.next().await {
                let progress = progress.map_err(|e| format!("Stream error: {}", e))?;
                
                if let (Some(completed), Some(total)) = (progress.completed, progress.total) {
                    let percentage = (completed as f64 / total as f64) * 100.0;
                    let completed_mb = completed as f64 / 1_048_576.0;
                    let total_mb = total as f64 / 1_048_576.0;
                    println!("{} - {:.1}% ({:.1}/{:.1} MB)", 
                             progress.status, percentage, completed_mb, total_mb);
                } else {
                    println!("{}", progress.status);
                }
            }

            println!("\nModel download completed!");
        }

        "info" => {
            if args.len() < 3 {
                println!("Usage: cargo run info <model_name>");
                println!("Example: cargo run info qwen3-coder:30b");
                return Ok(());
            }

            let model_name = &args[2];
            println!("Getting information for model: {}\n", model_name);

            let client = NaoriAI::ollama("http://localhost:11434".to_string(), model_name.to_string());
            let info = client.show_model_info(model_name).await?;

            println!("Model Information:");
            println!("- License: {}", info.license);
            println!("- Parameters: {}", info.parameters);
            println!("- Template length: {} characters", info.template.len());
            
            // Check tool support
            match client.supports_tool_calls().await {
                Ok(supports) => {
                    println!("- Tool calling support: {}", if supports { "Yes (native)" } else { "No (fallback available)" });
                }
                Err(_) => {
                    println!("- Tool calling support: Unknown");
                }
            }

            // Show template preview
            let preview = info.template.clone();
            println!("{}", preview);
        }

        _ => {
            println!("Unknown command: {}", command);
            println!("Available commands: list, pull, info, generate");
            println!("Use 'cargo run' without arguments for help.");
        }
    }

    Ok(())
}