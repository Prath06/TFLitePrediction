package com.example.ghantatest;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.example.ghantatest.ml.RiceTfLiteModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = (Button) findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                EditText ed1 = (EditText) findViewById(R.id.editTextNumberDecimal);
                EditText ed2 = (EditText) findViewById(R.id.editTextNumberDecimal2);
                EditText ed3 = (EditText) findViewById(R.id.editTextNumberDecimal3);
                EditText ed4 = (EditText) findViewById(R.id.editTextNumberDecimal4);
                EditText ed5 = (EditText) findViewById(R.id.editTextNumberDecimal5);


                String v1 = ed1.getText().toString();
                String v2 = ed2.getText().toString();
                String v3 = ed3.getText().toString();
                String v4 = ed4.getText().toString();
                String v5 = ed5.getText().toString();

                ByteBuffer byteBuffer = ByteBuffer.allocateDirect(5*5);
                byteBuffer.putFloat(Float.parseFloat(v1));
                byteBuffer.putFloat(Float.parseFloat(v2));
                byteBuffer.putFloat(Float.parseFloat(v3));
                byteBuffer.putFloat(Float.parseFloat(v4));
                byteBuffer.putFloat(Float.parseFloat(v5));

                try {
                    RiceTfLiteModel model = RiceTfLiteModel.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    RiceTfLiteModel.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    TextView textView = (TextView) findViewById(R.id.textView);
                    textView.setText(outputFeature0.getFloatArray()[0]+ "\n");

                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });


    }
}