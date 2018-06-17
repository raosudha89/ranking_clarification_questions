/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//package org.apache.lucene.demo;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.Math;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.FileVisitResult;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Date;
import java.util.stream.Stream;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

/** Simple command-line based search demo. */
public class SearchSimilarFiles {

  private SearchSimilarFiles() {}

  /** Simple command-line based search demo. */
  public static void main(String[] args) throws Exception, ParseException, IOException{
    String usage =
      "Usage:\tjava org.apache.lucene.demo.SearchSimilarFiles [-index dir] [-queries file] [-outputFile string]\n\n";
    if (args.length < 2 && ("-h".equals(args[0]) || "-help".equals(args[0]))) {
      System.out.println(usage);
      System.exit(0);
    }

    String index = "index";
    String field = "contents";
    String queriesPath = null;
    String queryString = null;
    String outputFile = null;   
 
    for(int i = 0;i < args.length;i++) {
      if ("-index".equals(args[i])) {
        index = args[i+1];
        i++;
      } else if ("-field".equals(args[i])) {
        field = args[i+1];
        i++;
      } else if ("-queries".equals(args[i])) {
        queriesPath = args[i+1];
        i++;
      } else if ("-outputFile".equals(args[i])) {
        outputFile = args[i+1];
        i++;
      }
    }
    IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(index)));
    IndexSearcher searcher = new IndexSearcher(reader);
    Analyzer analyzer = new StandardAnalyzer();

    QueryParser parser = new QueryParser(field, analyzer);
    PrintWriter writer = new PrintWriter(outputFile, "UTF-8");
      
    final Path queriesDir = Paths.get(queriesPath);
    if (Files.isDirectory(queriesDir)){
      Files.walkFileTree(queriesDir, new SimpleFileVisitor<Path>() {
        @Override
        public FileVisitResult visitFile(Path queryFile, BasicFileAttributes attrs) throws IOException {
          similarDocs(queryFile, parser, searcher, writer);
          return FileVisitResult.CONTINUE;
        }
      });
      //Stream<Path> queryFiles = Files.walk(queriesDir);
      //queryFiles.forEach(queryFile -> 
      //  similarDocs(queryFile, parser, searcher)
      //);
    }
    writer.close();
    reader.close();
  }

  static void similarDocs(Path file, QueryParser parser, IndexSearcher searcher, PrintWriter writer) {
    try (InputStream stream = Files.newInputStream(file)) {
      String filename = file.getFileName().toString();
      filename = filename.substring(0, filename.lastIndexOf('.'));
      writer.print(filename + ' ');
      String content = readDocument(new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8)));
      Query query = parser.parse(parser.escape(content));

      TopDocs results = searcher.search(query, 100);
      ScoreDoc[] hits = results.scoreDocs;

      int numTotalHits = results.totalHits;
      //System.out.println(numTotalHits + " total matching documents");

      for (int i = 0; i < Math.min(100, numTotalHits); i++) {
        Document doc = searcher.doc(hits[i].doc);
        String docname = doc.get("path");
        docname = docname.substring(0, docname.lastIndexOf('.'));
        //System.out.println("doc="+docname+" score="+hits[i].score);
        writer.print(docname + ' ');
      }
    } catch (IOException e) {
      e.printStackTrace();
    } catch (ParseException pe){
      System.out.println(pe.getStackTrace());
      System.out.println(pe.getMessage());
    } finally {
      writer.println();
    }
  }
 
  static String readDocument(BufferedReader br) throws IOException {
    StringBuilder sb = new StringBuilder();
    String line = br.readLine();

    while (line != null) {
      sb.append(line);
      sb.append(System.lineSeparator());
      line = br.readLine();
    }
    return sb.toString();
  }
}

