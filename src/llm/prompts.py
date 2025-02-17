import os
import logging
from typing import Any
import re

from langchain.prompts import (
    PromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

TEMPLATES_ROOT_PATH = "templates"

def _load_template(template_name: str) -> str:
    """
    Loads a template from a file.

    This function attempts to load a template file based on the given template name.
    It constructs the file path, reads the content, and returns it as a string.
    If the file is not found or any other error occurs during the process,
    appropriate exceptions are raised and logged.

    Args:
        template_name (str): The name of the template to load. This name is used
                             to construct the filename in the format
                             "template_{template_name}.txt".

    Returns:
        str: The content of the template file as a string.

    Raises:
        FileNotFoundError: If the template file does not exist at the expected path.
        Exception: For any other errors that occur during file reading or processing.

    Logs:
        - Info level log when the template is successfully loaded.
        - Error level log when the template file is not found or any other error occurs.
    """
    
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    
    try:
        with open(template_path, "r") as file:
            template = file.read()
        logging.info(f"Template {template_name} loaded successfully.")
        return template
    except FileNotFoundError:
        logging.error(f"Template file not found: {template_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading template {template_name}: {e}")
        raise

def _extract_input_variables(template: str) -> Any:
    """
    Extracts input variables from a template string.

    This function uses a regular expression to find all placeholders
    enclosed in curly braces within the template string.

    Args:
        template (str): The template string to extract variables from.

    Returns:
        Any: A list of extracted placeholders (input variables) found in the template.
              Each placeholder is represented as a string without the curly braces.

    Example:
        If the template is "Hello {name}, welcome to {place}!",
        the function will return ['name', 'place'].
    """
    pattern = r'\{(.*?)\}'
    placeholders = re.findall(pattern, template)
    return placeholders

def get_prompt(template_name: str = None, template: str = None) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate from either a template name or a provided template string.

    This function generates a ChatPromptTemplate, which can be used for structuring
    conversations with a language model. It either loads a template from a file using
    the provided template name or uses the directly provided template string.

    Args:
        template_name (str, optional): The name of the template file to load.
            If provided, the function will attempt to load the template from a file.
            Defaults to None.
        template (str, optional): The content of the template as a string.
            If provided, this will be used directly without loading from a file.
            Defaults to None.

    Returns:
        ChatPromptTemplate: A ChatPromptTemplate object that can be used to generate
        prompts for a conversation with a language model.

    Raises:
        ValueError: If neither template_name nor template is provided.

    Note:
        At least one of template_name or template must be provided. If both are
        provided, template_name takes precedence.
    """
    if template_name: # If template_name is provided, load the template
        template = _load_template(template_name)
    elif not template:
        raise ValueError("Either template_name or template must be provided.")
    
    input_variables = _extract_input_variables(template)
    
    human_message_prompt_template = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
            template=template,
            input_variables=input_variables,
        )
    )
    
    combined_prompt_template = ChatPromptTemplate.from_messages(
        [human_message_prompt_template]
    )
    
    return combined_prompt_template