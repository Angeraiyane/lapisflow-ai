import json
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. Taxonomy: ALL 77 Banking77 intents, each assigned to exactly one of 10 categories ---
CATEGORY_MAPPING = {
    "card_management": [
        "activate_my_card", "apple_pay_or_google_pay", "card_about_to_expire", "card_acceptance",
        "card_arrival", "card_delivery_estimate", "card_linking", "card_not_working", "card_swallowed",
        "contactless_not_working", "get_physical_card", "getting_spare_card", "order_physical_card",
        "visa_or_mastercard",
    ],
    "virtual_cards": [
        "disposable_card_limits", "get_disposable_virtual_card", "getting_virtual_card",
        "virtual_card_not_working",
    ],
    "unauthorized_transactions_security": [
        "card_payment_not_recognised", "cash_withdrawal_not_recognised", "compromised_card",
        "direct_debit_payment_not_recognised", "lost_or_stolen_card", "lost_or_stolen_phone",
    ],
    "account_and_identity": [
        "age_limit", "change_pin", "country_support", "edit_personal_details", "passcode_forgotten",
        "pin_blocked", "terminate_account", "unable_to_verify_identity", "verify_my_identity",
        "verify_source_of_funds", "why_verify_identity",
    ],
    "transfers": [
        "beneficiary_not_allowed", "cancel_transfer", "declined_transfer", "failed_transfer",
        "pending_transfer", "receiving_money", "transfer_into_account",
        "transfer_not_received_by_recipient", "transfer_timing",
    ],
    "top_up": [
        "automatic_top_up", "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit", "pending_top_up", "top_up_by_cash_or_cheque",
        "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card", "verify_top_up",
    ],
    "fees_charges": [
        "card_payment_fee_charged", "card_payment_wrong_exchange_rate", "cash_withdrawal_charge",
        "exchange_charge", "extra_charge_on_statement", "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge", "transaction_charged_twice", "transfer_fee_charged",
        "wrong_exchange_rate_for_cash_withdrawal",
    ],
    "fx_currency": [
        "exchange_rate", "exchange_via_app", "fiat_currency_support", "supported_cards_and_currencies",
    ],
    "atm_cash": [
        "atm_support", "declined_cash_withdrawal", "pending_cash_withdrawal", "wrong_amount_of_cash_received",
    ],
    "payments_and_refunds": [
        "Refund_not_showing_up", "declined_card_payment", "pending_card_payment",
        "request_refund", "reverted_card_payment?",
    ],
}

# Policy list (never auto-routed). Every intent here must exist in the dataset (checked below).
PROTECTED_INTENTS = [
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "compromised_card",
    "direct_debit_payment_not_recognised",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
]

# --- 2. Load Banking77 (CC BY 4.0, Casanueva et al., 2020) ---
URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
df = pd.concat([pd.read_csv(URL + "train.csv"), pd.read_csv(URL + "test.csv")], ignore_index=True)
df = df.rename(columns={"category": "intent"})
df["intent"] = df["intent"].str.strip()

# --- 3. Safety checks: no silent defaults ---
INTENT_TO_CATEGORY = {}
for cat, intents in CATEGORY_MAPPING.items():
    for intent in intents:
        assert intent not in INTENT_TO_CATEGORY, f"DUPLICATE intent in mapping: {intent}"
        INTENT_TO_CATEGORY[intent] = cat

real = set(df["intent"].unique())
assert real == set(INTENT_TO_CATEGORY), (
    f"Mapping mismatch. Unmapped real intents: {sorted(real - set(INTENT_TO_CATEGORY))} | "
    f"Names not in dataset: {sorted(set(INTENT_TO_CATEGORY) - real)}"
)
assert set(PROTECTED_INTENTS) <= real, f"Unknown protected intents: {set(PROTECTED_INTENTS) - real}"
assert len(real) == 77

df["category"] = df["intent"].map(INTENT_TO_CATEGORY)
df["is_protected"] = df["intent"].isin(PROTECTED_INTENTS).astype(int)

with open("data/categories.json", "w", encoding="utf-8") as f:
    json.dump({
        "categories": list(CATEGORY_MAPPING.keys()),
        "protected_intents": PROTECTED_INTENTS,
        "intent_to_category": INTENT_TO_CATEGORY,
    }, f, indent=2)

# --- 4. Stratified split on INTENT (keeps every protected intent in every split) ---
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["intent"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["intent"], random_state=42)

train_df.to_csv("data/train.csv", index=False, encoding="utf-8")
val_df.to_csv("data/val.csv", index=False, encoding="utf-8")
test_df.to_csv("data/test.csv", index=False, encoding="utf-8")

print("All 77 intents mapped exactly once. OK")
print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")
print("Category sizes (all data):")
print(df["category"].value_counts().to_string())
print(f"Protected cases  train={train_df.is_protected.sum()}  val={val_df.is_protected.sum()}  test={test_df.is_protected.sum()}")