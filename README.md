# Awesome Generative AI Data Scientist [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

A curated list of resources to help you become a Generative AI Data Scientist. This repository includes resources on building GenAI applications with Large Language Models (LLMs), and deploying LLMs and GenAI with Cloud-based solutions.

**NOTE** - This is a work in progress. Changes and additions are welcome. Please use Pull Requests to suggest modifications and improvements. 

## Contents:

- [Awesome Resources with Real World AI Use Cases](#awesome-resources-with-real-world-ai-use-cases)
- [Python Libraries](#python-libraries)
- [Projects (GenAI and LLMs)](#llm-deployment-cloud-services)
- [Courses and Training](#courses-and-training)

# Awesome Resources with Real World AI Use Cases

- [Awesome LLM Apps](https://github.com/Shubhamsaboo/awesome-llm-apps): LLM RAG AI Apps with Step-By-Step Tutorials
- [AI Data Science Team](https://github.com/business-science/ai-data-science-team): An AI-powered data science team of copilots that uses agents to help you perform common data science tasks 10X faster.
- [AI Hedge Fund](https://github.com/virattt/ai-hedge-fund): Proof of concept for an AI-powered hedge fund

# Python Libraries

## AI LLM Frameworks

- [LangChain](https://www.langchain.com/): A framework for developing applications powered by large language models (LLMs). [Documentation](https://python.langchain.com/) [Github](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph): A library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. [Documentation](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex](https://www.llamaindex.ai/): LlamaIndex is a framework for building context-augmented generative AI applications with LLMs. [Documentation](https://docs.llamaindex.ai/) [Github](https://github.com/run-llama/llama_index)
- [LlamaIndex Workflows](https://www.llamaindex.ai/blog/introducing-workflows-beta-a-new-way-to-create-complex-ai-applications-with-llamaindex): LlamaIndex workflows is a mechanism for orchestrating actions in the increasingly-complex AI application we see our users building.
- [CrewAI](https://www.crewai.com/): Streamline workflows across industries with powerful AI agents. [Documentation](https://docs.crewai.com/) [Github](https://github.com/crewAIInc/crewAI)

## LLM Models

- [OpenAI](https://github.com/openai/openai-python): The official Python library for the OpenAI API
- [Hugging Face Models](https://huggingface.co/models): Open LLM models by Meta, Mistral, and hundreds of other providers
- [Anthropic Claude](https://github.com/anthropics/anthropic-sdk-python): The official Python library for the Anthropic API
- [Meta Llama Models](https://llama.meta.com/): The open source AI model you can fine-tune, distill and deploy anywhere.
- [Google Gemini](https://github.com/google-gemini/generative-ai-python): The official Python library for the Google Gemini API
- [Ollama](https://github.com/ollama/ollama): Get up and running with large language models locally.
- [Grok](https://github.com/groq/groq-python): The official Python Library for the Groq API

## Vector Databases (RAG)

- [ChromaDB](https://github.com/chroma-core/chroma): The fastest way to build Python or JavaScript LLM apps with memory!
- [FAISS](https://github.com/facebookresearch/faiss): A library for efficient similarity search and clustering of dense vectors.
- [Qdrant](https://qdrant.tech/): High-Performance Vector Search at Scale
- [Pinecone](https://github.com/pinecone-io/pinecone-python-client): The official Pinecone Python SDK.
- [Milvus](https://github.com/milvus-io/milvus): Milvus is an open-source vector database built to power embedding similarity search and AI applications. 

## Useful Python GenAI Libraries

- [Embedchain](https://embedchain.ai/): Create an AI app on your own data in a minute [Documentation](https://docs.embedchain.ai/get-started/quickstart) [Github Repo](https://github.com/mem0ai/mem0/tree/main/embedchain)
- [Mem0](https://mem0.ai/): Mem0 is a self-improving memory layer for LLM applications, enabling personalized AI experiences that save costs and delight users. [Documentation](https://docs.mem0.ai/) [Github](https://github.com/mem0ai/mem0)
- [Docling by IBM](https://ds4sd.github.io/docling/): Parse documents and export them to the desired format with ease and speed. [Github](https://github.com/DS4SD/docling)
- [Markitdown by Microsoft](https://github.com/microsoft/markitdown): Python tool for converting files and office documents to Markdown.
- [Gitingest](https://gitingest.com/): Turn any Git repository into a simple text ingest of its codebase. This is useful for feeding a codebase into any LLM. [Github](https://github.com/cyclotruc/gitingest)

# LLM Deployment (Cloud Services)

- [AWS Bedrock](https://aws.amazon.com/bedrock/): Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon
- [Microsoft Azure AI Services](https://azure.microsoft.com/en-us/products/ai-services): Azure AI services help developers and organizations rapidly create intelligent, cutting-edge, market-ready, and responsible applications with out-of-the-box and prebuilt and customizable APIs and models. 
- [Google Vertex AI](https://cloud.google.com/vertex-ai): Vertex AI is a fully-managed, unified AI development platform for building and using generative AI.
- [NVIDIA NIM](https://www.nvidia.com/en-us/ai): NVIDIA NIM™, part of NVIDIA AI Enterprise, provides containers to self-host GPU-accelerated inferencing microservices for pretrained and customized AI models across clouds, data centers, and workstations.


# Projects (GenAI and LLMs)

## Cookbooks and Examples:

- [LangChain Cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/README.md): Example code for building applications with LangChain, with an emphasis on more applied and end-to-end examples.
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples): Example code for building applications with LangGraph
- [Llama Index Examples](https://github.com/run-llama/llama_index/tree/main/docs/docs/examples): Example code for building applications with Llama Index
- [Streamlit LLM Examples](https://github.com/streamlit/llm-examples): Streamlit LLM app examples for getting started

## Cloud Examples and Cookbooks

### Amazon Web Services (AWS)

- [Azure Generative AI Examples](https://github.com/Azure/azureml-examples/tree/main/sdk/python/generative-ai): Prompt Flow and RAG Examples for use with the Microsoft Azure Cloud platform
- [Amazon Bedrock Workshop](https://github.com/aws-samples/amazon-bedrock-workshop): Introduces how to leverage foundation models (FMs) through Amazon Bedrock

### Google Cloud Platform (GCP)

- [Google Vertex AI Examples](https://github.com/GoogleCloudPlatform/vertex-ai-samples): Notebooks, code samples, sample apps, and other resources that demonstrate how to use, develop and manage machine learning and generative AI workflows using Google Cloud Vertex AI
- [Google Generative AI Examples](https://github.com/GoogleCloudPlatform/generative-ai): Sample code and notebooks for Generative AI on Google Cloud, with Gemini on Vertex AI

### NVIDIA 

- [NVIDIA NIM Anywhere](https://github.com/NVIDIA/nim-anywhere): An entrypoint for developing with NIMs that natively scales out to full-sized labs and up to production environments.
- [NVIDIA NIM Deploy](https://github.com/NVIDIA/nim-deploy): Reference implementations, example documents, and architecture guides that can be used as a starting point to deploy multiple NIMs and other NVIDIA microservices into Kubernetes and other production deployment environments.


# Courses and Training

## Free AI For Data Scientists Workshop

Get free training on how to build and deploy Generative AI / ML Solutions. [Register for our next free workshop here.](https://learn.business-science.io/ai-register)

## 8-Week AI Bootcamp by Business Science

We are launching a new Generative AI Bootcamp that covers end-to-end AI application development and Cloud deployment. [Find out more about how to build AI with Python, and attend our free AI training session here.](https://learn.business-science.io/registration-ai-workshop-2)
