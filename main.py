import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import uvicorn


load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class TravelFormData(BaseModel):
    destination: str
    departureCity: str
    departureDate: str
    returnDate: str
    flightBudget: str
    accommodationBudget: str
    tripType: Literal["leisure", "business", "adventure", "romantic", "family", "solo"]
    numberOfPeople: Literal["1", "2", "3", "4", "5", "6+"]
    rentCar: bool
    needsFlight: bool 
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)

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

chain = prompt_template | llm

@app.post("/travel-plan", response_class=PlainTextResponse)
async def create_travel_plan(form_data: TravelFormData):
    try:
        llm_input = {
            "destination": form_data.destination,
            "departureCity": form_data.departureCity,
            "departureDate": form_data.departureDate,
            "returnDate": form_data.returnDate,
            "flightBudget": form_data.flightBudget,
            "accommodationBudget": form_data.accommodationBudget,
            "tripType": form_data.tripType, 
            "numberOfPeople": form_data.numberOfPeople,
            "rentCar": "Yes" if form_data.rentCar else "No",  
            "needsFlight": "Yes" if form_data.needsFlight else "No", 
        }

        response = await chain.ainvoke(llm_input)

        generated_text = response.content
        cleaned_text = generated_text.strip() 

        return cleaned_text
    except Exception as e:
        print(f"Error processing travel plan request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.get("/")
async def read_root():
    return {"message": "TravelPlan FastAPI backend is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)