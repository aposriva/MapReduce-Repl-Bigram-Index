import java.io.IOException;
import java.util.HashMap;
import java.util.Arrays;
import java.util.HashSet;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class BigramIndex {
  public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    private HashSet<String> selected_bigrams = new HashSet<>(Arrays.asList(
        "computer science",
        "information retrieval",
        "power politics",
        "los angeles",
        "bruce willis"));

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString().toLowerCase().replaceAll("[^a-z0-9]+", " "));
      Text docID = new Text(itr.nextToken().strip());
      String nextWord = "";
      String nextNextWord = "";
      String newBigram = "";
      while (itr.hasMoreTokens()) {
        if (nextWord.isBlank()) {
          nextWord = itr.nextToken().replaceAll("[^a-z]+", " ").strip();
          continue;
        }
        if (!itr.hasMoreTokens()) {
          continue;
        }
        nextNextWord = itr.nextToken().replaceAll("[^a-z]+", " ").strip();
        if (nextNextWord.isBlank()) {
          nextWord = nextNextWord;
          continue;
        }
        newBigram = nextWord + " " + nextNextWord;
        if (selected_bigrams.contains(newBigram)) {
          word.set(newBigram);
          context.write(word, docID);
        }
        nextWord = nextNextWord;
      }
    }
  }

  public static class IntSumReducer extends Reducer<Text, Text, Text, Text> {
    private Text result = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      HashMap<String, Integer> doc_freq = new HashMap<>();
      for (Text val : values) {
        int sum = 1;
        String docID = val.toString();
        // System.out.println("value = "+ docID);
        if (doc_freq.containsKey(docID)) {
          sum = doc_freq.get(docID) + 1;
        }
        doc_freq.put(docID, sum);
      }
      String resultStr = "";
      for (String docID : doc_freq.keySet()) {
        if (docID.contains(":")) {
          resultStr += docID + " ";
        } else {
          resultStr += docID + ":" + doc_freq.get(docID) + " ";
        }
      }
      result.set(resultStr);
      // System.out.println("reduce called = "+ key.toString() + " " + doc_freq.size()
      // + " " + doc_freq.keySet().size() + " " + resultStr);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "Bigram Index");

    job.setJarByClass(BigramIndex.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(Text.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}// BigramIndex
