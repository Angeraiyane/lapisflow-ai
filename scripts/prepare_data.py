import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Définition des 10 catégories et de la liste protégée
PROTECTED_INTENTS = [
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "compromised_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone"
]

CATEGORY_MAPPING = {
    "card_issues": [
        "card_about_to_expire", "card_acceptance", "card_arrival", "card_delivery_estimate",
        "card_linking", "card_not_working", "card_swallowed", "getting_spare_card",
        "getting_virtual_card", "order_physical_card", "pin_blocked", "replacement_card_duration"
    ],
    "unauthorized_transactions_security": [
        "card_payment_not_recognised", "cash_withdrawal_not_recognised",
        "compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone"
    ],
    "account_management": [
        "age_limit", "change_pin", "close_account", "country_support", "edit_personal_details",
        "passcode_forgotten", "verify_my_identity", "verify_source_of_funds", "verify_top_up",
        "why_verify_identity"
    ],
    "transfers_payments": [
        "beneficiary_not_allowed", "cancel_transfer", "failed_transfer", "pending_transfer",
        "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
        "transfer_timing", "virtual_card_not_working"
    ],
    "top_up": [
        "add_money", "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
        "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "reverted_card_payment?",
        "supported_cards_and_currencies", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
        "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted"
    ],
    "fees_charges": [
        "card_payment_fee_charged", "cash_withdrawal_charge", "exchange_charge",
        "extra_charge_on_statement", "transaction_charged_twice", "transfer_fee_charged"
    ],
    "atm_cash": [
        "atm_support", "automatic_top_up", "wrong_amount_of_cash_received"
    ],
    "exchange_crypto": [
        "disposable_cards_exceed_limits", "exchange_rate", "exchange_via_app", "fiat_currency_exchange"
    ],
    "refunds_reversals": [
        "request_refund", "refund_timing", "declined_card_payment", "declined_cash_withdrawal", "declined_transfer"
    ],
    "general_inquiries": [
        "activate_my_card", "apple_pay_or_google_pay", "contactless_not_working",
        "direct_debit_payment_problem", "receiving_money", "terminate_account", "unable_to_verify_identity"
    ]
}

INTENT_TO_CATEGORY = {}
for cat, intents in CATEGORY_MAPPING.items():
    for intent in intents:
        INTENT_TO_CATEGORY[intent] = cat

# 2. Sauvegarde du fichier de configuration de taxonomie
metadata = {
    "categories": list(CATEGORY_MAPPING.keys()),
    "protected_intents": PROTECTED_INTENTS,
    "intent_to_category": INTENT_TO_CATEGORY
}

with open("data/categories.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(" [1/3] data/categories.json généré avec succès.")

# 3. Téléchargement direct des CSV officiels de Banking77 via GitHub
print("Téléchargement direct des données Banking77...")
url_train = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
url_test = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

df_train_raw = pd.read_csv(url_train)
df_test_raw = pd.read_csv(url_test)
df = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

# Renommage des colonnes si besoin (la colonne cible s'appelle 'category' dans ce dépôt)
if "category" in df.columns and "intent" not in df.columns:
    df.rename(columns={"category": "intent"}, inplace=True)

df["category"] = df["intent"].apply(lambda i: INTENT_TO_CATEGORY.get(str(i).strip(), "general_inquiries"))
df["is_protected"] = df["intent"].apply(lambda i: 1 if str(i).strip() in PROTECTED_INTENTS else 0)

# 4. Stratified Split (Train: 70%, Val: 15%, Test: 15%)
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["category"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["category"], random_state=42)

train_df.to_csv("data/train.csv", index=False, encoding="utf-8")
val_df.to_csv("data/val.csv", index=False, encoding="utf-8")
test_df.to_csv("data/test.csv", index=False, encoding="utf-8")

print(f" [2/3] Données divisées : Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
print(f" [3/3] Cas protégés dans le test : {test_df['is_protected'].sum()} cas.")