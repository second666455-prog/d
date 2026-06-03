import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# تحميل البيانات
df = pd.read_csv('/home/ubuntu/upload/NDX4mGKiq5Ug6iHaVyDJz4jlv9qHqTNT(1).csv')

# 1. التعامل مع القيم المفقودة
# للمسافة المقطوعة (Mileage_KM): سنستخدم المتوسط
imputer_num = SimpleImputer(strategy='mean')
df['Mileage_KM'] = imputer_num.fit_transform(df[['Mileage_KM']])

# لنوع الوقود (Fuel_Type): سنستخدم القيمة الأكثر تكراراً (mode)
imputer_cat = SimpleImputer(strategy='most_frequent')
df['Fuel_Type'] = imputer_cat.fit_transform(df[['Fuel_Type']]).ravel()

# 2. حذف الأعمدة غير الضرورية
# Car_ID لا يؤثر على السعر
df.drop('Car_ID', axis=1, inplace=True)

# 3. تحويل البيانات النصية إلى أرقام (Label Encoding)
# سنقوم بحفظ الـ encoders لاستخدامها لاحقاً في الـ Deployment
label_encoders = {}
for col in ['Brand', 'Fuel_Type', 'Transmission']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# حفظ البيانات المنظفة
df.to_csv('/home/ubuntu/cleaned_data.csv', index=False)
print("Data Preprocessing Complete. Cleaned data saved to /home/ubuntu/cleaned_data.csv")

# عرض أول 5 أسطر من البيانات بعد المعالجة
print(df.head())
