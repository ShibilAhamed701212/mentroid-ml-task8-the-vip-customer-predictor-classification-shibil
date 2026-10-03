"""Dummy customer-support messages with the route a human agent would choose."""

SAMPLE_MESSAGES: list[tuple[str, str]] = [
    ("I was charged twice for my subscription this month, please refund one payment.", "Billing"),
    ("Can I get a copy of my invoice from last March?", "Billing"),
    ("My credit card expired, how do I update my payment method?", "Billing"),
    ("Why did my monthly bill go up from $10 to $15?", "Billing"),
    ("I cancelled my plan but you still took money from my account.", "Billing"),
    ("The app crashes every time I try to upload a photo.", "Technical Support"),
    ("I can't log in, it keeps saying 'invalid token' even after resetting my password.", "Technical Support"),
    ("Your website shows a 500 error when I click checkout.", "Technical Support"),
    ("The sync between my phone and laptop stopped working after the last update.", "Technical Support"),
    ("Videos keep buffering and the audio is out of sync.", "Technical Support"),
    ("What are your customer service opening hours?", "General Inquiry"),
    ("Do you ship to Canada?", "General Inquiry"),
    ("I'd like to know more about your company's privacy policy.", "General Inquiry"),
    ("Are you planning to open a store in Dubai?", "General Inquiry"),
    ("Who should I contact about a partnership opportunity?", "General Inquiry"),
]

# Written after the route descriptions were tuned on SAMPLE_MESSAGES. Not used to tune
# descriptions or threshold, though it did motivate the catch-all design (see classifier.py).
HELDOUT_MESSAGES: list[tuple[str, str]] = [
    ("Is there a student discount on the annual plan?", "Billing"),
    ("I need a receipt with my company's VAT number on it.", "Billing"),
    ("The free trial ended and I was billed without any warning.", "Billing"),
    ("Push notifications stopped arriving on my Android phone.", "Technical Support"),
    ("The export to PDF button does nothing when I press it.", "Technical Support"),
    ("I get a 'connection timed out' message whenever I open the dashboard.", "Technical Support"),
    ("Where is your head office located?", "General Inquiry"),
    ("Do you have any job openings for designers?", "General Inquiry"),
    ("What languages is your product available in?", "General Inquiry"),
]
