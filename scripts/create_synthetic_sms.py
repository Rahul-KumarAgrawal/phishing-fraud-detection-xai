"""
create_synthetic_sms.py
=======================
Creates a minimal SYNTHETIC SMS dataset for pipeline testing ONLY.
This is NOT a real dataset and must NOT be used for research or evaluation.

Purpose: Allow the full training/evaluation pipeline to be demonstrated
         before the real Mendeley dataset is downloaded.

The real dataset must be downloaded from:
    https://data.mendeley.com/datasets/f45bkkt8pr/1
and placed at data/raw/SMS_dataset.csv

Usage (from repo root):
    python scripts/create_synthetic_sms.py
"""

import pathlib
import pandas as pd

RAW_DIR = pathlib.Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT = RAW_DIR / "SMS_dataset.csv"

if OUT.exists():
    print(f"File already exists: {OUT}")
    print("Delete it first if you want to regenerate the synthetic dataset.")
    raise SystemExit(0)

# ── Synthetic examples (for pipeline testing only) ────────────────────────
SMISHING = [
    "URGENT: Your bank account has been suspended. Verify immediately at http://bit.ly/secure-verify",
    "Congratulations! You've won $1000 gift card. Click http://prize-winner.xyz/claim to redeem NOW",
    "Your package is on hold. Update delivery preferences: http://dhl-parcel.ru/update?id=AX92",
    "FREE entry to win £250 Primark voucher, text WIN to 80080 now",
    "ALERT: Unusual login detected. Secure your account: http://paypal-secure.phish.cc/confirm",
    "Your account will be closed. Click here to verify: http://192.168.1.1/login?session=abc",
    "You have a pending tax refund of $450. Claim at http://irs-gov-refund.tk/claim",
    "Your netflix subscription expired! Renew now: http://bit.ly/netflix-renew-2024",
    "WINNER! As a valued customer you've been selected. Call 0800-555-3456 now for your reward",
    "Your Apple ID is locked. Verify: http://apple-id-verify.ru/login?code=X7B2",
    "Bank: We notice suspicious activity. Please call 0800-000-0000 or visit http://hsbc-secure.xyz",
    "Smishing: Your parcel requires £2.50 customs fee. Pay: http://royal-mail-fee.cc/pay",
    "FREE msg: You have won a 2-week holiday to Benidorm worth £5000. To claim txt HOLIDAY to 85023",
    "HSBC: Your account has been compromised. Verify at http://hsbc-alert.ru",
    "Your PayPal account needs verification. Login: http://paypal.phishing-site.com/verify",
    "Urgent! Your debit card PIN has been suspended. Call 01223900150 immediately",
    "You are a winner! Please call 09061702893 from a landline. Your reference code: TC4563899",
    "Claim your free lottery prize by texting WIN to 82277",
    "Amazon: Your payment failed. Update billing at http://amazon-billing.xyz/update",
    "FREE ringtone: text POLY to 8007",
    "You have been approved for a £2000 loan. Click http://quick-loans.phish.cc/apply",
    "IMPORTANT: Your Google account will be deleted. Verify now: http://google-verify.cc/secure",
    "Your shipment is awaiting customs clearance. Pay £1.99: http://fedex-customs.ru",
    "We tried to call you. Call back 09066350750 to claim your £300 prize.",
    "Congratulations! ur awarded 2 cinema ticket. Call 09061209465",
    "Win a £1000 cash prize or a prize worth £5000 Text WIN to 80085 now",
    "FREE PRIZE CLAIM: Reply WIN to claim your prize now! Expires 31/01",
    "NOTICE: Your account will be suspended. Act now: http://microsoftsupport.phish.cc",
    "You have qualified for our £1000 prize! To claim call 09064012160 from a landline",
    "Your credit card was charged $499. Dispute: http://bank-refund.cc/dispute?id=TX9942",
]

HAM = [
    "Hey, are you coming to the party tonight? Let me know!",
    "Meeting is rescheduled to 3pm on Thursday. See you there.",
    "Can you pick up some milk on your way home?",
    "I'll be 10 minutes late. Start without me.",
    "Happy birthday! Hope you have a wonderful day.",
    "Did you watch the match last night? What a game!",
    "The report is ready for your review. Please check your email.",
    "Lunch at the usual place? 12:30 works for me.",
    "Don't forget the team standup at 10am tomorrow.",
    "Thanks for your help yesterday. Really appreciated it!",
    "Your appointment is confirmed for Monday 9am.",
    "I've sent you the files. Let me know if you need anything else.",
    "Call me when you get a chance, nothing urgent.",
    "The project deadline has been extended to next Friday.",
    "Can we reschedule our call? I have a conflict.",
    "Great work on the presentation today!",
    "Heading to the gym after work. Want to join?",
    "The movie starts at 7:30. Should we grab dinner first?",
    "I'm at the supermarket. Do you need anything?",
    "Just landed. Will be home in about an hour.",
    "The office will be closed on Monday for the bank holiday.",
    "Reminder: dentist appointment tomorrow at 2pm.",
    "Your order has shipped. Estimated delivery: Thursday.",
    "Hi, I got your message. I'll reply properly tonight.",
    "The train is delayed by 15 minutes.",
    "Can you forward me the minutes from yesterday's meeting?",
    "The kids got full marks on their tests today!",
    "Dinner reservation confirmed for Saturday at 7pm, party of 4.",
    "I think I left my umbrella at your place. Can I pick it up?",
    "The weather looks great this weekend. Fancy a hike?",
    "Your prescription is ready for collection.",
    "Good news – the bug has been fixed in the latest release.",
    "Let's catch up for coffee soon. It's been ages!",
    "Quick question: did you manage to sort out the invoice issue?",
    "The quarterly report looks strong. Well done to the whole team.",
]

rows = (
    [{"LABEL": "smishing", "TEXT": t} for t in SMISHING]
    + [{"LABEL": "ham", "TEXT": t} for t in HAM]
)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"Created SYNTHETIC SMS dataset: {OUT}")
print(f"  smishing: {len(SMISHING)}")
print(f"  ham:      {len(HAM)}")
print()
print("WARNING: This is a SYNTHETIC dataset for pipeline testing only.")
print("Replace data/raw/SMS_dataset.csv with the real Mendeley dataset")
print("before reporting any evaluation metrics.")
