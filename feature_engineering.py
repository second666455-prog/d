import pandas as pd

# تحميل البيانات المنظفة
df = pd.read_csv('/home/ubuntu/cleaned_data.csv')

# إضافة ميزة جديدة: عمر السيارة (Car_Age)
# نفترض أن السنة الحالية هي 2024 (أو 2026 كما في النظام)
current_year = 2026
df['Car_Age'] = current_year - df['Model_Year']

# إضافة ميزة: متوسط المسافة المقطوعة سنوياً (KM_per_Year)
# لتجنب القسمة على صفر، نضيف 1 لعمر السيارة
df['KM_per_Year'] = df['Mileage_KM'] / (df['Car_Age'] + 1)

# حفظ البيانات بعد هندسة المميزات
df.to_csv('/home/ubuntu/featured_data.csv', index=False)
print("Feature Engineering Complete. Featured data saved to /home/ubuntu/featured_data.csv")
print(df.head())
