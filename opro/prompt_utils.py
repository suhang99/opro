# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The utility functions for prompting GPT and Google Cloud models."""

import time
from llama import Llama, Dialog
from typing import List, Optional
import google.generativeai as palm
import openai
import os


def call_openai_server_single_prompt(
    prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with an input string."""
  try:
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_decode_steps,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content

  except openai.error.Timeout as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.RateLimitError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"API error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIConnectionError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"API connection error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.ServiceUnavailableError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Service unavailable. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except OSError as e:
    retry_time = 5  # Adjust the retry time as needed
    print(
        f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
    )
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )


def call_openai_server_func(
    inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with a list of input strings."""
  if isinstance(inputs, str):
    inputs = [inputs]
  outputs = []
  for input_str in inputs:
    output = call_openai_server_single_prompt(
        input_str,
        model=model,
        max_decode_steps=max_decode_steps,
        temperature=temperature,
    )
    outputs.append(output)
  return outputs


def call_palm_server_from_cloud(
    input_text, model="text-bison-001", max_decode_steps=20, temperature=0.8
):
  """Calling the text-bison model from Cloud API."""
  assert isinstance(input_text, str)
  assert model == "text-bison-001"
  all_model_names = [
      m
      for m in palm.list_models()
      if "generateText" in m.supported_generation_methods
  ]
  model_name = all_model_names[0].name
  try:
    completion = palm.generate_text(
        model=model_name,
        prompt=input_text,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
    )
    output_text = completion.result
    return [output_text]
  except:  # pylint: disable=bare-except
    retry_time = 10  # Adjust the retry time as needed
    print(f"Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_palm_server_from_cloud(
        input_text, max_decode_steps=max_decode_steps, temperature=temperature
    )


class LlamaModel:
  def __init__(self, model="llama-2-7b-chat", ckpt_dir="", tokenizer_path="", max_seq_len=512, max_batch_size=8) -> None:
    """Initializing the llama model"""
    assert model == "llama-2-7b-chat"
    assert os.path.exists(ckpt_dir)
    assert os.path.exists(tokenizer_path)
    self.ckpt_dir = ckpt_dir
    self.tokenizer_path = tokenizer_path
    self.max_seq_len = max_seq_len
    self.max_batch_size = max_batch_size

  def create_model(self, temperature=0.6, max_decode_steps=20):
    self.temperature = temperature
    self.max_decode_steps = max_decode_steps
    self.generator = Llama.build(ckpt_dir=self.ckpt_dir, tokenizer_path=self.tokenizer_path, max_seq_len=self.max_seq_len, max_batch_size=self.max_batch_size)

  def call_llama(self, input_text):
    """Calling the llama model"""
    assert isinstance(input_text, str)

    dialogs: List[Dialog] = [[{"role":"user", "content": input_text}]]
    results = self.generator.chat_completion(dialogs, temperature=self.temperature, top_p=0.9)
    assert len(results) == 1
    return results[0]['generation']['content']