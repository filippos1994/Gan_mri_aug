


visits_list = ["s1/e1", "s1/e2", "s2/e1", "s2/e2", "s2/e3"]
name_list = []

for visit in visits_list:
	name_list.append(visit.split('/')[0])

print(name_list)

name_list = list(set(name_list))
print(name_list)
