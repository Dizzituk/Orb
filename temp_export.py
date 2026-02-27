import sqlite3, json
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')
data = {}

data['work_logs'] = []
for r in conn.execute("SELECT work_date,tour_id,delivery_count,collections,stops,attempted,done,total_hours,net_hours,rate_per_parcel,gross_earnings,per_hour,tax_year FROM finance_daily_work_logs ORDER BY work_date"):
    data['work_logs'].append({'date':r[0],'tour':r[1],'deliveries':r[2],'collections':r[3],'stops':r[4],'attempted':r[5],'done':r[6],'total_hours':r[7],'net_hours':r[8],'rate':r[9],'earnings':r[10],'per_hour':r[11],'tax_year':r[12]})

data['expenses'] = []
for r in conn.execute("SELECT t.transaction_date,t.description,t.amount,t.expense_scope,t.is_tax_deductible,t.deductible_amount,c.display_name,c.hmrc_category,t.notes FROM finance_transactions t LEFT JOIN finance_categories c ON t.category_id=c.id WHERE t.transaction_type='expense' AND t.is_deleted=0 ORDER BY t.transaction_date"):
    data['expenses'].append({'date':r[0],'description':r[1],'amount':r[2],'scope':r[3],'deductible':r[4],'deductible_amount':r[5],'category':r[6] or 'Uncategorised','hmrc_category':r[7],'notes':r[8]})

data['cat_totals'] = []
for r in conn.execute("SELECT COALESCE(c.display_name,'Uncategorised'),c.hmrc_category,COUNT(*),SUM(t.amount),SUM(t.deductible_amount) FROM finance_transactions t LEFT JOIN finance_categories c ON t.category_id=c.id WHERE t.transaction_type='expense' AND t.is_deleted=0 GROUP BY COALESCE(c.display_name,'Uncategorised') ORDER BY SUM(t.amount) DESC"):
    data['cat_totals'].append({'category':r[0],'hmrc_category':r[1],'count':r[2],'total':r[3],'deductible':r[4]})

data['income'] = []
for r in conn.execute("SELECT transaction_date,description,amount,expense_scope FROM finance_transactions WHERE transaction_type='income' AND is_deleted=0 ORDER BY transaction_date"):
    data['income'].append({'date':r[0],'description':r[1],'amount':r[2],'scope':r[3]})

wl = conn.execute("SELECT COUNT(*),SUM(delivery_count),SUM(gross_earnings),AVG(delivery_count),AVG(gross_earnings),SUM(net_hours) FROM finance_daily_work_logs").fetchone()
inc = conn.execute("SELECT SUM(amount) FROM finance_transactions WHERE transaction_type='income' AND is_deleted=0").fetchone()
exp = conn.execute("SELECT SUM(amount),SUM(deductible_amount) FROM finance_transactions WHERE transaction_type='expense' AND is_deleted=0").fetchone()
data['summary'] = {'total_days':wl[0],'total_deliveries':wl[1],'total_earnings':round(wl[2],2),'avg_deliveries':round(wl[3],1),'avg_earnings':round(wl[4],2),'total_hours':wl[5],'total_income_bank':inc[0] or 0,'total_expenses':round(exp[0] or 0,2),'total_deductible':round(exp[1] or 0,2)}

conn.close()
with open(r'D:\Orb\temp_tax_data.json','w') as f:
    json.dump(data,f,default=str)
print(f"{len(data['work_logs'])} logs, {len(data['expenses'])} expenses, {len(data['cat_totals'])} cats, {len(data['income'])} income")

