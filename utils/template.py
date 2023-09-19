from langchain import PromptTemplate


# PlayGround
def inference_template(prompt, coordinates=None):         
    prompt_template = PromptTemplate.from_template(
        """
            user's prompt: {prompt}
            coordinates: {coordinates}
            
            Please use English when using the tools input.
            Especially when using the 'grounded_sam' tool, the classes to be included in the 'class_list' must be in English.
            
            And respond should be Korean.           
        """
    )
    
    prompt = prompt_template.format(prompt = prompt,
                                    coordinates=coordinates)
    return prompt


def image_generate_template(prompt, num_images):         
    prompt_template = PromptTemplate.from_template(
        """
            prompt: {prompt}
            num_images: {num_images}
            
            Please use English when using the tools input.            
            And respond should be "None".           
        """
    )
    
    prompt = prompt_template.format(prompt = prompt,
                                    num_images=num_images)
    return prompt