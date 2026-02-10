# Lead Discovery Python Agent (Smart Medium x Leads Gorilla Prototype)

Developed by **Rohun Kulshrestha** for **Smart Medium**, this prototype represents an intelligent, AI-driven lead generation system designed to help businesses uncover new clients with data-backed precision.

## Overview

The **Lead Discovery Agent** uses the **Google Places API** and **machine learning techniques** to autonomously:
- Discover local businesses by category and location  
- Extract and analyze online presence data (websites, contact info, SEO indicators)  
- Perform **sentiment analysis** on customer reviews using **VADER NLP**  
- Compute an **AI-driven lead score (0–100)** that ranks sales opportunities by potential

This forms the foundation for Smart Medium’s integration with **Leads Gorilla**, expanding its AI capabilities in **lead intelligence and automation**.

---

## Setup Instructions

### 1: Clone the repository
git clone https://github.com/YOUR_USERNAME/leads-gorilla-prototype.git
cd leads-gorilla-prototype

### 2: Clone the repository
python -m venv venv
venv\Scripts\activate   # on Windows
# or
source venv/bin/activate  # on Mac/Linux

### 3: Install Dependencies
pip install -r requirements.txt

### 4: Create .env file
GOOGLE_API_KEY=your_google_api_key_here

### 5:Run
python main.py --category " " --location " "
