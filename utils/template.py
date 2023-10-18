from langchain import PromptTemplate


# PlayGround
def image_editor_template(prompt):         
    prompt_template = PromptTemplate.from_template(
        """
            prompt: {prompt}
            
            Make sure prompt must be in English when using the tool.
        """
    )
    
    prompt = prompt_template.format(prompt=prompt)
    return prompt


def image_generate_template(prompt, num_images):         
    prompt_template = PromptTemplate.from_template(
        """
            input_prompt: {prompt}
            num_images: {num_images}
            
            The input_prompt must be in English.           
        """
    )
    
    prompt = prompt_template.format(prompt = prompt,
                                    num_images=num_images)
    return prompt