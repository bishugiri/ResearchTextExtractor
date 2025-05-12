from countryguess import guess_country
import pycountry

def test_country_guess():
    # Test cases with different inputs
    test_cases = [
        "France",        # Exact match
        "Fraunce",      # Close misspelling
        "Frence",       # Another misspelling
        "Franxx",       # Very different
        "Deutschland",  # Foreign name
        "Deutchland",  # Misspelled foreign name
        "USA",         # Common abbreviation
        "U.S.A.",     # With periods
        "United States of America",  # Full name
        "xyz"          # Invalid input
    ]
    
    print("\nTesting countryguess package:")
    print("-" * 50)
    
    for test_input in test_cases:
        result = guess_country(test_input)
        print(f"\nInput: {test_input}")
        print(f"Result: {result}")
        if result:
            print(f"Detected Country: {result.get('name_short', 'N/A')}")
            print(f"ISO Code: {result.get('iso3', 'N/A')}")
        else:
            print("No match found")

if __name__ == "__main__":
    test_country_guess() 