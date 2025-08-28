import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse # Changed from HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal # Import Literal for specific string values
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS to allow requests from your Next.js frontend
origins = [
    "http://localhost:3000",  # For local development
    "https://v0-create-it-eight.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming request data
class TravelFormData(BaseModel):
    destination: str
    departureCity: str
    departureDate: str
    returnDate: str
    flightBudget: str
    accommodationBudget: str
    # Use Literal types to match frontend's Select component values
    tripType: Literal["leisure", "business", "adventure", "romantic", "family", "solo"]
    numberOfPeople: Literal["1", "2", "3", "4", "5", "6+"]
    rentCar: bool
    needsFlight: bool # Added needsFlight as per frontend
    # Removed 'currency' field as it's not present in the frontend's formData

# Initialize the Generative AI model
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

# Define the prompt template for the AI agent
prompt_template = PromptTemplate(
    input_variables=[
        "destination", "departureCity", "departureDate", "returnDate",
        "flightBudget", "accommodationBudget", "numberOfPeople", "rentCar", "tripType", "needsFlight"
    ],
    template="""Act as a Travel AI Agent that provides personalized recommendations for the best possible flights, accommodations, and car rentals based on the user's inputs. Also, create a detailed daily itinerary with specific suggestions on things to do, see, eat, and experience in the destination, tailored to the travel type (e.g., make it family-friendly with kid-safe activities if tripType is 'family', adventurous with outdoor pursuits if 'adventure', etc.). Ensure all recommendations respect the budgets, number of people, and preferences provided.

User Inputs:
- Destination: {destination}
- Departure City: {departureCity}
- Departure Date: {departureDate}
- Return Date: {returnDate}
- Flight/Transportation Budget: INR {flightBudget} total for {numberOfPeople} people (Note: If 'Needs Flight' is No, this is for other transportation)
- Accommodation Budget: INR {accommodationBudget} per night for {numberOfPeople} people
- Rent a Car: {rentCar}
- Number of People: {numberOfPeople}
- Travel Type: {tripType}
- Needs Flight: {needsFlight}

For recommendations:
- Flights/Transportation: Suggest 2-3 direct or efficient options with estimated costs within the budget. If 'Needs Flight' is No, suggest train/bus/car options. Include placeholder links like 'Book on Kayak.com' or 'Check Expedia for details'.
- Accommodations: Suggest 2-3 hotels/airbnbs/apartments within the accommodation budget, considering the number of people (e.g., family suites). Include placeholder links like 'Book on Booking.com'.
- Car Rentals: If requested (Rent a Car is 'Yes'), suggest 2-3 options with daily rates. Include placeholder links like 'Rent via Rentalcars.com'. If not (Rent a Car is 'No'), recommend public transport or rideshares.
- Itinerary: Create a day-by-day plan from departure to return, including:
  - Morning/afternoon/evening activities with specific suggestions (e.g., visit landmarks, try local cuisine, relax at beaches).
  - Tailor to travel type: For family, include parks/museums; for romantic, sunset spots/dinners; etc.
  - Factor in travel time, rest days, and budget-friendly tips.
  - Suggest free or low-cost activities to balance budgets.

Return the output in a clean, readable plain text format. Use clear headings and bullet points. Do NOT use HTML tags, markdown formatting (like `**` or `#`), or JSON. Just provide the structured text.
"""
)

# Create the LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

@app.post("/travel-plan", response_class=PlainTextResponse) # Changed response_class to PlainTextResponse
async def create_travel_plan(form_data: TravelFormData):
    try:
        # Prepare data for the LLMChain
        llm_input = {
            "destination": form_data.destination,
            "departureCity": form_data.departureCity,
            "departureDate": form_data.departureDate,
            "returnDate": form_data.returnDate,
            "flightBudget": form_data.flightBudget,
            "accommodationBudget": form_data.accommodationBudget,
            "tripType": form_data.tripType, # Directly use tripType
            "numberOfPeople": form_data.numberOfPeople, # Directly use numberOfPeople
            "rentCar": "Yes" if form_data.rentCar else "No",  # Convert boolean to string for prompt
            "needsFlight": "Yes" if form_data.needsFlight else "No", # Convert boolean to string for prompt
        }

        # Invoke the LLMChain to get the response
        response = await llm_chain.ainvoke(llm_input)

        # The response from LLMChain is a dictionary, extract the 'text' key
        generated_text = response.get('text', '')

        # No HTML cleaning needed if the prompt is adjusted to return plain text
        # However, some basic newline handling might still be useful if the LLM adds extra ones
        cleaned_text = generated_text.strip() # Remove leading/trailing whitespace

        return cleaned_text
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing travel plan request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Add a root endpoint for health check or basic info
@app.get("/")
async def read_root():
    return {"message": "TravelPlan FastAPI backend is running!"}

