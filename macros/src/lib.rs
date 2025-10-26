extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use serde_json::json;
use syn::{ItemFn, Pat, PatType, Type, parse_macro_input};

fn rust_type_to_json_type(ty: &Type) -> &'static str {
    if let Type::Path(type_path) = ty {
        if type_path.path.is_ident("String") {
            return "string";
        } else if type_path.path.is_ident("i32")
            || type_path.path.is_ident("i64")
            || type_path.path.is_ident("usize")
        {
            return "integer";
        } else if type_path.path.is_ident("f32") || type_path.path.is_ident("f64") {
            return "number";
        } else if type_path.path.is_ident("bool") {
            return "boolean";
        }
    }
    "string"
}

#[proc_macro_attribute]
pub fn tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let tool_func_name = format_ident!("{}_tool", func_name);

    let mut description = String::new();
    for attr in &func.attrs {
        if attr.path().is_ident("doc") {
            if let Ok(name_value) = attr.meta.require_name_value() {
                if let syn::Expr::Lit(expr_lit) = &name_value.value {
                    if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                        description.push_str(&lit_str.value().trim());
                        description.push(' ');
                    }
                }
            }
        }
    }
    let description = description.trim().to_string();

    let mut params_properties = serde_json::Map::new();
    let mut required_params = Vec::new();
    let mut arg_names = Vec::new();
    let mut arg_names_str = Vec::new();
    let mut arg_types = Vec::new();

    for input in &func.sig.inputs {
        if let syn::FnArg::Typed(PatType { pat, ty, .. }) = input {
            if let Pat::Ident(pat_ident) = &**pat {
                let arg_name = pat_ident.ident.to_string();
                let json_type = rust_type_to_json_type(ty);

                params_properties.insert(
                    arg_name.clone(),
                    json!({
                        "type": json_type,
                        "description": ""
                    }),
                );
                required_params.push(arg_name.clone());
                arg_names.push(pat_ident.ident.clone());
                arg_names_str.push(arg_name.clone());
                arg_types.push(ty.clone());
            }
        }
    }

    let parameters_json = json!({
        "type": "object",
        "properties": params_properties,
        "required": required_params
    })
    .to_string();

    let expanded = quote! {
        pub fn #tool_func_name() -> naori_ai::Tool {
            #func

            naori_ai::Tool {
                name: #func_name_str.to_string(),
                description: #description.to_string(),
                parameters: serde_json::from_str(#parameters_json).unwrap(),
                function: Box::new(|args| {
                    #(let #arg_names: #arg_types = serde_json::from_value(args[#arg_names_str].clone()).unwrap();)*
                    #func_name(#(#arg_names),*).to_string()
                }),
            }
        }
    };

    TokenStream::from(expanded)
}
