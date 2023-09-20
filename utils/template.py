from langchain import PromptTemplate


# PlayGround
def inference_template(prompt):         
    prompt_template = PromptTemplate.from_template(
        """
            input_prompt: {prompt}
            
            The input_prompt must be in English.
            Especially when using the 'grounded_sam' tool, the classes to be included in the 'class_list' must be in English.
            
            And respond should be Korean.           
        """
    )
    
    prompt = prompt_template.format(prompt = prompt)
    return prompt


def image_generate_template(prompt, num_images):         
    prompt_template = PromptTemplate.from_template(
        """
            input_prompt: {prompt}
            num_images: {num_images}
            
            The input_prompt must be in English.        
            And respond should be "None".           
        """
    )
    
    prompt = prompt_template.format(prompt = prompt,
                                    num_images=num_images)
    return prompt