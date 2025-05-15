import json
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Standards data
standards = [
    {
        "id": "FAS 4",
        "title_en": "Currency Translation",
        "title_ar": "ترجمة العملات",
        "description_en": "Standard for recording foreign currency transactions and translating financial statements.",
        "description_ar": "معيار لتسجيل معاملات العملات الأجنبية وترجمة البيانات المالية."
    },
    {
        "id": "FAS 7",
        "title_en": "Investments in Real Estate",
        "title_ar": "الاستثمارات العقارية",
        "description_en": "Standard for accounting for investments in properties for rental or capital appreciation.",
        "description_ar": "معيار محاسبة الاستثمارات في العقارات للتأجير أو زيادة رأس المال."
    },
    {
        "id": "FAS 10",
        "title_en": "Istisna'a and Parallel Istisna'a",
        "title_ar": "الاستصناع والاستصناع الموازي",
        "description_en": "Standard for manufacturing contracts where payment is made in installments.",
        "description_ar": "معيار لعقود التصنيع حيث يتم الدفع على أقساط."
    },
    {
        "id": "FAS 28",
        "title_en": "Ijarah and Ijarah Muntahia Bittamleek",
        "title_ar": "الإجارة والإجارة المنتهية بالتمليك",
        "description_en": "Standard for lease agreements and leases ending with ownership transfer.",
        "description_ar": "معيار لاتفاقيات الإيجار والإيجارات المنتهية بنقل الملكية."
    },
    {
        "id": "FAS 32",
        "title_en": "Investment Agency (Al-Wakala Bi Al-Istithmar)",
        "title_ar": "وكالة الاستثمار (الوكالة بالاستثمار)",
        "description_en": "Standard for investment agency relationships between investors and agents.",
        "description_ar": "معيار لعلاقات وكالة الاستثمار بين المستثمرين والوكلاء."
    }
]

# Examples data
examples = [
    {
        "standard_id": "FAS 4",
        "title_en": "Converting USD to EUR for international trade",
        "title_ar": "تحويل الدولار الأمريكي إلى اليورو للتجارة الدولية",
        "scenario_en": "Al Baraka Bank needs to record a transaction where they purchased machinery from a European supplier:\n- Purchase price: €500,000\n- Exchange rate on purchase date: 1 EUR = 1.10 USD\n- Exchange rate on payment date (30 days later): 1 EUR = 1.12 USD\nHow should this be recorded in the books?",
        "scenario_ar": "يحتاج بنك البركة إلى تسجيل معاملة شراء آلات من مورد أوروبي:\n- سعر الشراء: 500,000 يورو\n- سعر الصرف في تاريخ الشراء: 1 يورو = 1.10 دولار أمريكي\n- سعر الصرف في تاريخ الدفع (بعد 30 يومًا): 1 يورو = 1.12 دولار أمريكي\nكيف يجب تسجيل هذا في الدفاتر؟"
    },
    {
        "standard_id": "FAS 7",
        "title_en": "Acquiring an office building for rental",
        "title_ar": "الاستحواذ على مبنى مكتبي للتأجير",
        "scenario_en": "Islamic Finance House purchased a commercial building:\n- Purchase price: $5,000,000\n- Legal fees: $50,000\n- Building improvements: $200,000\n- Expected rental income: $400,000 per year\nHow should this investment be recorded and measured?",
        "scenario_ar": "اشترى بيت التمويل الإسلامي مبنى تجاري:\n- سعر الشراء: 5,000,000 دولار\n- الرسوم القانونية: 50,000 دولار\n- تحسينات المبنى: 200,000 دولار\n- الدخل المتوقع من الإيجار: 400,000 دولار سنويًا\nكيف يجب تسجيل وقياس هذا الاستثمار؟"
    },
    {
        "standard_id": "FAS 10",
        "title_en": "Manufacturing contract for custom equipment",
        "title_ar": "عقد تصنيع لمعدات مخصصة",
        "scenario_en": "Al Salam Bank entered into an Istisna'a contract with a manufacturer:\n- Contract value: $1,000,000 for custom manufacturing equipment\n- Payment schedule: 30% upfront, 30% halfway, 40% upon delivery\n- Manufacturing period: 8 months\nHow should the bank record this transaction?",
        "scenario_ar": "دخل بنك السلام في عقد استصناع مع مصنّع:\n- قيمة العقد: 1,000,000 دولار لمعدات تصنيع مخصصة\n- جدول الدفع: 30٪ مقدمًا، 30٪ في المنتصف، 40٪ عند التسليم\n- فترة التصنيع: 8 أشهر\nكيف يجب على البنك تسجيل هذه المعاملة؟"
    },
    {
        "standard_id": "FAS 28",
        "title_en": "Leasing equipment with ownership transfer",
        "title_ar": "تأجير معدات مع نقل الملكية",
        "scenario_en": "Alpha Islamic Bank entered into an Ijarah MBT for a generator:\n- Generator cost: $450,000\n- Import tax and freight: $42,000\n- Lease term: 2 years\n- Annual rental: $300,000\n- Purchase option at end: $3,000\nHow should Alpha Bank record this transaction?",
        "scenario_ar": "دخل بنك ألفا الإسلامي في إجارة منتهية بالتمليك لمولد كهربائي:\n- تكلفة المولد: 450,000 دولار\n- ضريبة الاستيراد والشحن: 42,000 دولار\n- مدة الإيجار: سنتان\n- الإيجار السنوي: 300,000 دولار\n- خيار الشراء في النهاية: 3,000 دولار\nكيف يجب على بنك ألفا تسجيل هذه المعاملة؟"
    },
    {
        "standard_id": "FAS 32",
        "title_en": "Investment agency relationship",
        "title_ar": "علاقة وكالة استثمار",
        "scenario_en": "Qatar Islamic Bank accepted $10M from investors on Wakala basis:\n- Expected profit rate: 5% annually\n- Bank's agency fee: 20% of profits above 5%\n- Investment term: 1 year\n- Actual return achieved: 7%\nHow should this be accounted for?",
        "scenario_ar": "قبل بنك قطر الإسلامي 10 مليون دولار من المستثمرين على أساس الوكالة:\n- معدل الربح المتوقع: 5٪ سنويًا\n- رسوم وكالة البنك: 20٪ من الأرباح فوق 5٪\n- مدة الاستثمار: سنة واحدة\n- العائد الفعلي المحقق: 7٪\nكيف يجب المحاسبة عن ذلك؟"
    }
]

# Glossary data
glossary = [
    {
        "term": "Ijarah",
        "definition_en": "A lease contract where one party transfers the right to use an asset to another party for an agreed period at an agreed consideration.",
        "definition_ar": "عقد إيجار حيث ينقل طرف حق استخدام أصل إلى طرف آخر لفترة متفق عليها بمقابل متفق عليه."
    },
    {
        "term": "Murabaha",
        "definition_en": "A sales contract where the seller expressly mentions the cost incurred on the sold commodity and sells it to another person by adding some profit.",
        "definition_ar": "عقد بيع حيث يذكر البائع صراحةً التكلفة التي تكبدها على السلعة المباعة ويبيعها لشخص آخر بإضافة بعض الربح."
    },
    {
        "term": "Wakala",
        "definition_en": "An agency contract where one party appoints another party to act on their behalf for a specific task.",
        "definition_ar": "عقد وكالة حيث يعين طرف طرفًا آخر للتصرف نيابة عنه لمهمة محددة."
    },
    {
        "term": "Istisna'a",
        "definition_en": "A contract of sale where a commodity is transacted before it comes into existence, requiring the manufacturer to make it with payment from the buyer either in advance or by installments.",
        "definition_ar": "عقد بيع حيث يتم تداول سلعة قبل وجودها، مما يتطلب من المصنع صنعها مع دفع المشتري إما مقدمًا أو على أقساط."
    },
    {
        "term": "Sukuk",
        "definition_en": "Islamic financial certificates, similar to bonds, that comply with Shariah law.",
        "definition_ar": "شهادات مالية إسلامية، مشابهة للسندات، تتوافق مع الشريعة الإسلامية."
    }
]

# Write data to JSON files
with open("data/standards.json", "w", encoding="utf-8") as f:
    json.dump(standards, f, ensure_ascii=False, indent=2)

with open("data/examples.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

with open("data/glossary.json", "w", encoding="utf-8") as f:
    json.dump(glossary, f, ensure_ascii=False, indent=2)

print("Data files have been populated successfully!")
