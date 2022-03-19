package kmeans;

import java.io.IOException;
import java.io.*;
import java.net.URI;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;



public class KMeans {
	
    public static class Map extends Mapper<LongWritable, Text, IntWritable, Text> {
    
        @Override
        protected void map(LongWritable key, Text value, Context context)
    			throws IOException, InterruptedException {
    		    		
    		Configuration conf = (Configuration) context.getConfiguration() ;
    		int k = Integer.valueOf(conf.get("k")) ;
    		double min = Double.MAX_VALUE ;
    		int best = -1 ;
    		String[] all_dim = value.toString().split(",") ;
    		for ( int i = 0;i < k; i ++) {
    			String center = conf.get("centroid" + Integer.toString(i)) ;
    			String[] center_values = center.split(",") ;
    			double dist = 0 ;
    			for(int j = 0; j < all_dim.length - 1; j++) {
    				double point = Double.parseDouble(all_dim[j]) ;
    				double center_value = Double.parseDouble(center_values[j]) ;
    				dist += Math.pow(point - center_value, 2) ; 
    			}
    			dist = Math.sqrt(dist);
    			if(dist < min) {
    				min = dist; 
    				best = i;
    			}
    		}
    		context.write(new IntWritable(best) , value);
    	}
    }
    public static class Reduce extends Reducer<IntWritable, Text, IntWritable, Text> {
    	@Override
    	protected void reduce(IntWritable centroid_ind, Iterable<Text> data,Context context) throws IOException, InterruptedException {
    		
    		Configuration conf = (Configuration) context.getConfiguration() ;
    		int dim = Integer.valueOf(conf.get("dim")) ;
    		double[] avg = new double[dim] ; 
    		int count = 0 ;
    		for(Text d : data) {
    			String[] elements = d.toString().split(",") ;
    			for(int i = 0 ; i < elements.length -1  ; i++) {
    				double val = Double.valueOf(elements[i]) ;
       				avg[i] +=  val  ;
    			}
    			count++ ;
    		}
    		for(int i = 0 ; i < dim  ; i++) {
    			avg[i] /= count;
			}
    		StringBuilder s = new StringBuilder() ;
    		for(int i = 0 ; i < avg.length ; i++) {
    			s.append(String.valueOf(avg[i])) ;
    			if( i != avg.length - 1) s.append(",") ;
    		}
    		String value = s.toString() ;
    		context.write(centroid_ind, new Text(value));    		
    	}
    }
    
    public static void main(String[] args) throws Exception {
        run(args);
    }
    public static void run(String[] args) throws Exception {
    	if(args.length != 4) {
			System.out.print("Invalid Error!") ;
			System.exit(-1);
		}
    	String input = args[0], output = args[1];
		int k = Integer.valueOf(args[2]),  dim = Integer.valueOf(args[3]) ;
		System.out.println("***********************************");
        String centroids = "/irisInput/centers.txt";
        boolean isdone = false;
        double[][] old_centers = new double[k][dim] ;
		int iteration = 1 ;
        while (isdone == false) {
        	Job job = Job.getInstance() ;
			Configuration conf = job.getConfiguration() ;
			
			String path = centroids ;
			Configuration temp = new Configuration();
			FileSystem file = FileSystem.get(URI.create(path), temp); 
			Path input_path = new Path(path);
			FSDataInputStream in = file.open(input_path);
			BufferedReader buffer = new BufferedReader(new InputStreamReader(in));
			double[][] new_centers = new double[k][dim] ;
			conf.set("k", Integer.toString(k));
			conf.set("dim", Integer.toString(dim));
			for(int i = 0 ; i < k ; i++) {	
				String line = buffer.readLine() ;
				System.out.println("Centroid");
				System.out.println(line);
				int key = Integer.valueOf(line.split("\t")[0]) ;
				String[] center = line.split("\t")[1].split(",") ;
				if(center.length != dim ){
					System.out.print("Invalid Input Length!");
					System.exit(-1);
				}
				for(int j = 0 ; j < dim ; j++) {
					new_centers[key][j] = Double.valueOf(center[j]) ;
					
				}
				conf.set("centroid" + key, line.split("\t")[1]);
			}
			
			double error = 0 ;
			for(int i=0;i<k;i++){
				for(int j=0;j<dim;j++){
					error += Math.pow(new_centers[i][j] - old_centers[i][j], 2) ;
				}
			}
			double tolerance = 0.000001;
			if(error < tolerance)
				break ;
        	
			job.setJarByClass(KMeans.class);
        	job.setJobName("KMeans");
        	job.setMapOutputKeyClass(IntWritable.class);
        	job.setMapOutputValueClass(Text.class);
        	job.setOutputKeyClass(IntWritable.class);
        	job.setOutputValueClass(Text.class);
            job.setMapperClass(Map.class);
            job.setReducerClass(Reduce.class);
			FileInputFormat.addInputPath(job, new Path(input));
			FileOutputFormat.setOutputPath(job, new Path(output+"_"+Integer.toString(iteration)));
            job.waitForCompletion(true);
            
            old_centers = new_centers;
			centroids =  output + "_" + Integer.toString(iteration) + "/part-r-00000";
			iteration++ ;
        }

    }
}