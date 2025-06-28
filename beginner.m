%%KNN ile cinsiyet tahmini

%Boy-kilo =>Features
X=[170 65;160 55;175 55;158 50;180 85;162 52];

%Etiketler
Y=["Erkek";"Kadın";"Erkek";"Kadın";"Erkek";"Kadın"];

%K-NN sınıflandırma modeli oluşturma
Mdl=fitcknn(X,Y,"NumNeighbors",3);

new_data=[168 80];
predicted=predict(Mdl,new_data);
disp("Tahmin edilen cinsiyet: "+predicted);