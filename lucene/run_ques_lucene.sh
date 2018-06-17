SITE_DIR=$1
#/usr/lib/jvm/java-1.8.0-oracle.x86_64/bin/javac -cp 'lucene-6.3.0/libs/*' IndexFiles.java

#/usr/lib/jvm/java-1.8.0-oracle.x86_64/bin/java -cp '.:lucene-6.3.0/libs/*' IndexFiles -docs $SITE_DIR/ques_docs/ -index $SITE_DIR/ques_doc_indices/

/usr/lib/jvm/java-1.8.0-oracle-1.8.0.151-1jpp.5.el7.x86_64/bin/java -cp '.:lucene-6.3.0/libs/*' IndexFiles -docs $SITE_DIR/ques_docs/ -index $SITE_DIR/ques_doc_indices/

#/usr/lib/jvm/java-1.8.0-oracle.x86_64/bin/javac -cp 'lucene-6.3.0/libs/*' SearchSimilarFiles.java

#/usr/lib/jvm/java-1.8.0-oracle.x86_64/bin/java -cp '.:lucene-6.3.0/libs/*' SearchSimilarFiles -index $SITE_DIR/ques_doc_indices/ -queries $SITE_DIR/ques_docs/ -outputFile $SITE_DIR/lucene_similar_questions.txt

/usr/lib/jvm/java-1.8.0-oracle-1.8.0.151-1jpp.5.el7.x86_64/bin/java -cp '.:lucene-6.3.0/libs/*' SearchSimilarFiles -index $SITE_DIR/ques_doc_indices/ -queries $SITE_DIR/ques_docs/ -outputFile $SITE_DIR/lucene_similar_questions.txt
