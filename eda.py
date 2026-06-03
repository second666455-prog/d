import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# إعداد الخطوط لدعم اللغة العربية إذا لزم الأمر، لكن هنا سنركز على البيانات
# تحميل البيانات
df = pd.read_csv('/home/ubuntu/upload/NDX4mGKiq5Ug6iHaVyDJz4jlv9qHqTNT(1).csv')

# معلومات عامة
print("--- Data Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Unique Values in Categorical Columns ---")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].unique()}")

# رسم بعض المخططات للفهم
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True)
plt.title('Price Distribution')
plt.savefig('/home/ubuntu/price_distribution.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Mileage_KM', y='Price', hue='Brand')
plt.title('Price vs Mileage')
plt.savefig('/home/ubuntu/price_vs_mileage.png')
