#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "rashid";
const char* password = "12345678";

WebServer server(80);

int led1 = 15;
int led2 = 13;

void handleRoot() {

  String page = "<html><body>";
  page += "<h1>ESP32 LED Control</h1>";

  page += "<p>LED 1 (GPIO15)</p>";
  page += "<a href='/led1on'><button>ON</button></a>";
  page += "<a href='/led1off'><button>OFF</button></a>";

  page += "<p>LED 2 (GPIO13)</p>";
  page += "<a href='/led2on'><button>ON</button></a>";
  page += "<a href='/led2off'><button>OFF</button></a>";

  page += "</body></html>";

  server.send(200, "text/html", page);
}

void led1on(){
  digitalWrite(led1, HIGH);
  handleRoot();
}

void led1off(){
  digitalWrite(led1, LOW);
  handleRoot();
}

void led2on(){
  digitalWrite(led2, HIGH);
  handleRoot();
}

void led2off(){
  digitalWrite(led2, LOW);
  handleRoot();
}

void setup() {

  Serial.begin(115200);

  pinMode(led1, OUTPUT);
  pinMode(led2, OUTPUT);

  WiFi.begin(ssid, password);

  Serial.print("Connecting");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi Connected");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/led1on", led1on);
  server.on("/led1off", led1off);
  server.on("/led2on", led2on);
  server.on("/led2off", led2off);

  server.begin();
}

void loop() {
  server.handleClient();
}
