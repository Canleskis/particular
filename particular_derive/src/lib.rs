extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn;

#[proc_macro_derive(Particle)]
pub fn particle_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    
    impl_particle(&ast)
}

fn impl_particle(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = &ast.generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics Particle for #name #ty_generics #where_clause {
            fn position(&self) -> Vec3 {
                self.position
            }
        
            fn mu(&self) -> f32 {
                self.mu
            }
        }
    };

    gen.into()
}