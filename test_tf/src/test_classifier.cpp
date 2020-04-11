#include "tf_utils.hpp"
#include <scope_guard.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <dirent.h>
#include <string>
#define IMG_SIZE 32

using namespace std;

int convert_pos(int x,int y,int c, int row_size = IMG_SIZE, int col_size = IMG_SIZE){
  return c*row_size*col_size + y*row_size + x;
}

class Point{
public:
  Point( int _x = 0, int _y = 0):x(_x),y(_y){
  }

  Point get_avg(Point b){
    return Point( (b.x+x/2), (b.y+y/2));
  }

  int x;
  int y;
};

void fix_center( const int& g_size_x, const int& g_size_y, Point& c){
  if(g_size_x < IMG_SIZE || g_size_y < IMG_SIZE){
    cout<<"error img size"<<endl;
    return;
  }
  c.x = min(c.x,g_size_x-IMG_SIZE);
  c.y = min(c.y,g_size_y-IMG_SIZE);
  c.x = max(c.x, IMG_SIZE);
  c.y = max(c.y, IMG_SIZE);
}

void gen_image( const Point& c,const Point& bb_ll,const Point& bb_ur, 
                vector<float>& img, const vector<float>& cur_data,int& g_size_x,int& g_size_y){
  int ll_x = c.x - IMG_SIZE/2, ll_y = c.y - IMG_SIZE/2;
  
  for(int i=0; i<IMG_SIZE; ++i){
    for(int j=0; j<IMG_SIZE; ++j){
      img[convert_pos(i, j, 0)] = cur_data[convert_pos(ll_x+i, ll_y+j, 0, g_size_x, g_size_y)];
      img[convert_pos(i, j, 1)] = cur_data[convert_pos(ll_x+i, ll_y+j, 1, g_size_x, g_size_y)]; 
      img[convert_pos(i, j, 2)] = cur_data[convert_pos(ll_x+i, ll_y+j, 2, g_size_x, g_size_y)];

      if(ll_x+i > bb_ur.x || ll_x+i < bb_ll.x)
        img[convert_pos(i, j, 0)] = 0;
          
      if(ll_y+i > bb_ur.y || ll_y+i < bb_ll.y)
        img[convert_pos(i, j, 1)] = 0;   
    }
  }
}

void read_input( vector<vector<float>> &data_set, vector<int>& labels){
    DIR *dir;
    FILE * fp;
    struct dirent *ent;
    string file_dir = "../bin/";
    Point bb_ll, bb_ur;
    Point p1, p2, center;
    int g_size_x, g_size_y;
    int label;
    char s1[10];
    vector<float> cur_data, img;
    int count = 0;

    img.resize(IMG_SIZE*IMG_SIZE*3);

    if ((dir = opendir(file_dir.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
          if(count == 100)
            break;
            // cout<<"parsing data "<<count<<endl;
          count++;
          string file_name(ent->d_name);
          if (file_name.compare(".")==0 || file_name.compare("..")==0) continue;
          file_name = file_dir + file_name;
          fp = fopen(file_name.c_str(), "r");
          
          fscanf(fp,"%s%d%d", s1, &g_size_x, &g_size_y);
          cur_data.resize(g_size_x*g_size_y*3);

          fscanf(fp,"%s%d%d%d%d", s1, &bb_ll.x, &bb_ll.y, &bb_ur.x, &bb_ur.y);

          fscanf(fp,"%s%d%d%d%d", s1, &p1.x, &p1.y, &p2.x, &p2.y);
          cur_data[convert_pos(p1.x, p1.y, 2, g_size_x, g_size_y)] = 1;
          cur_data[convert_pos(p2.x, p2.y, 2, g_size_x, g_size_y)] = 1;


          for(int i=0; i<g_size_x; ++i){
              for(int j=0; j<g_size_y; ++j){
                int x, y, x_cost, y_cost;
                fscanf(fp,"%s%d%d%d%d", s1, &x, &y, &x_cost, &y_cost);
                cur_data[convert_pos( x, y, 0, g_size_x, g_size_y)] = x_cost;
                cur_data[convert_pos( x, y, 1, g_size_x, g_size_y)] = y_cost;
                cur_data[convert_pos( x, y, 2, g_size_x, g_size_y)] = 0;
              }
          }

          fscanf(fp,"%s%d", s1, &label);

          center = p1.get_avg(p2);

          fix_center(g_size_x, g_size_y, center);
          gen_image( center, bb_ll, bb_ur, img, cur_data, g_size_x, g_size_y);
          labels.emplace_back(label);
          data_set.emplace_back(img);
          fclose(fp);
        }
        
        closedir (dir);
    } else {
        cout<<"wrong dir\n";
        return;
    }
}

int main() {
  vector<vector<float>> data_set;
  vector<int> labels;
  vector<vector<bool>> is_train;
  vector<TF_Tensor*> in_tensors;
  vector<TF_Output> in_ops;
  bool tmp_train = false;

  read_input(data_set, labels);

  is_train.resize(data_set.size());
  for(int i=0; i<data_set.size(); ++i)
    is_train[i].emplace_back(false);
  
  cout<<"total samples "<<data_set.size()<<endl;

  auto graph = tf_utils::LoadGraph("graph.pb");
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph); }; // Auto-delete on scope exit.
  if (graph == nullptr) {
    cout << "Can't load graph" << endl;
    return 1;
  }

  auto input_data = TF_Output{TF_GraphOperationByName(graph, "Placeholder"), 0};
  if (input_data.oper == nullptr) {
    cout << "Can't init input_data" << endl;
    return 2;
  }
  const vector<int64_t> input_data_dims = {1, IMG_SIZE, IMG_SIZE, 3};
  auto input_data_tensor = tf_utils::CreateTensor(TF_FLOAT, input_data_dims, data_set[0]);
  SCOPE_EXIT{ tf_utils::DeleteTensor(input_data_tensor); }; // Auto-delete on scope exit.
  in_ops.emplace_back(input_data);
  in_tensors.emplace_back(input_data_tensor);


  auto input_isTrain = TF_Output{TF_GraphOperationByName(graph, "Placeholder_3"), 0};
  if (input_isTrain.oper == nullptr) {
    cout << "Can't init input_isTrain" << endl;
    return 2; 
  }
  const vector<int64_t> input_isTrain_dims = {1, 1};
  auto input_isTrain_tensor = tf_utils::CreateTensor(TF_BOOL, input_isTrain_dims.data(), input_isTrain_dims.size(), &tmp_train, sizeof(bool));
  SCOPE_EXIT{ tf_utils::DeleteTensor(input_isTrain_tensor); }; // Auto-delete on scope exit.
  in_ops.emplace_back(input_isTrain);
  in_tensors.emplace_back(input_isTrain_tensor);



  auto out_op = TF_Output{TF_GraphOperationByName(graph, "classification_model/dropout_2/dropout/Identity"), 0};
  if (out_op.oper == nullptr) {
    cout << "Can't init out_op" << endl;
    return 3;
  }

  TF_Tensor* output_tensor = nullptr;
  SCOPE_EXIT{ tf_utils::DeleteTensor(output_tensor); }; // Auto-delete on scope exit.

  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.
  auto options = TF_NewSessionOptions();
  auto sess = TF_NewSession(graph, options, status);
  TF_DeleteSessionOptions(options);

  if (TF_GetCode(status) != TF_OK) {
    return 4;
  }

  TF_SessionRun(sess,
                nullptr, // Run options.
                in_ops.data(), in_tensors.data(), 2, // Input tensors, input tensor values, number of inputs.
                &out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                nullptr, 0, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );
  cout<<"end session run\n";
  if (TF_GetCode(status) != TF_OK) {
    cout << "Error run session "<<TF_GetCode(status)<<"\n";
    return 5;
  }

  TF_CloseSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    cout << "Error close session";
    return 6;
  }

  TF_DeleteSession(sess, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Error delete session";
    return 7;
  }

  auto data = static_cast<float*>(TF_TensorData(output_tensor));

  std::cout << "Output vals: " << data[0] << ", " << data[1] << std::endl;

  return 0;
}
